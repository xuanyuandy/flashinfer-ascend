"""Real A5 execution: no mocked hardware, FP8 primitive or CPU fallback."""

import pytest

import flashinfer as finpu


@pytest.fixture(scope="module")
def torch_a5():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or finpu.current_hardware().soc_model != "Ascend950":
        pytest.skip("requires an Ascend950 NPU")
    return torch


@pytest.mark.parametrize("a_dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("b_dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("out_dtype", ["float16", "bfloat16"])
def test_bmm_fp8(torch_a5, a_dtype, b_dtype, out_dtype):
    torch = torch_a5
    torch.manual_seed(10)
    a = (torch.randn(3, 8, 128) * 10).to(getattr(torch, a_dtype))
    b = (torch.randn(3, 64, 128) * 10).to(getattr(torch, b_dtype)).transpose(-1, -2)
    sa, sb = torch.tensor([0.01]), torch.tensor([0.02])
    dtype = getattr(torch, out_dtype)
    out = torch.empty(3, 8, 64, dtype=dtype, device="npu")
    result = finpu.bmm_fp8(a.npu(), b.npu(), sa.npu(), sb.npu(), dtype, out=out)
    expected = finpu.bmm_fp8(a, b, sa, sb, dtype, backend="reference")
    assert result is out
    torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=1e-2)


def test_fp8_dot_product_does_not_overflow_before_scaling(torch_a5):
    torch = torch_a5
    a = torch.full((2, 4, 128), 448.0).to(torch.float8_e4m3fn)
    b = torch.full((2, 128, 64), 448.0).to(torch.float8_e4m3fn)
    scale = torch.tensor([1e-3])
    actual = finpu.bmm_fp8(a.npu(), b.npu(), scale.npu(), scale.npu(), torch.float16)
    expected = finpu.bmm_fp8(a, b, scale, scale, torch.float16, backend="reference")
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["K", "MN"])
@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float8_e5m2"])
def test_groupwise_fp8(torch_a5, mode, dtype_name):
    torch = torch_a5
    torch.manual_seed(11)
    dtype = getattr(torch, dtype_name)
    a = (torch.randn(32, 256) * 10).to(dtype)
    b = (torch.randn(128, 256) * 10).to(dtype)
    sa = torch.rand(32, 2) * 0.02 + 0.01
    sb = torch.rand(1, 2) * 0.02 + 0.01
    if mode == "MN":
        sa, sb = sa.t().contiguous(), sb.t().contiguous()
    kwargs = dict(scale_major_mode=mode, scale_granularity_mnk=(1, 128, 128))
    actual = finpu.gemm_fp8_nt_groupwise(a.npu(), b.npu(), sa.npu(), sb.npu(), **kwargs)
    expected = finpu.gemm_fp8_nt_groupwise(a, b, sa, sb, backend="reference", **kwargs)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("block_scale", [False, True])
def test_fp8_moe_with_empty_experts(torch_a5, block_scale):
    torch = torch_a5
    torch.manual_seed(12)
    seq, hidden, inter, experts = 8, 256, 256, 4
    # Only experts 0 and 1 receive tokens; the other groups are empty.
    logits = torch.randn(seq, experts)
    logits[:, 2:] = -100
    bias = torch.zeros(experts)
    x = (torch.randn(seq, hidden) * 0.1).to(torch.float8_e4m3fn)
    w1 = (torch.randn(experts, 2 * inter, hidden) * 0.05).to(torch.float8_e4m3fn)
    w2 = (torch.randn(experts, hidden, inter) * 0.05).to(torch.float8_e4m3fn)
    out = torch.empty(seq, hidden, dtype=torch.bfloat16, device="npu")
    expected = torch.empty(seq, hidden, dtype=torch.bfloat16)
    kwargs = dict(
        num_experts=experts, top_k=2, intermediate_size=inter,
        local_expert_offset=0, local_num_experts=experts, routing_method_type=0,
    )
    if block_scale:
        xs = torch.rand(hidden // 128, seq) + 0.5
        s1 = torch.rand(experts, 2 * inter // 128, hidden // 128) + 0.5
        s2 = torch.rand(experts, hidden // 128, inter // 128) + 0.5
        args = (logits, bias, x, xs, w1, s1, w2, s2)
        fn = finpu.fp8_block_scale_moe
        kwargs.update(n_groups=1, top_k_groups=1, routed_scaling=1.0)
    else:
        s1 = torch.rand(experts) + 0.5
        s1_gate = torch.rand(experts) + 0.5
        s2 = torch.rand(experts) + 0.5
        args = (logits, bias, x, w1, s1, s1_gate, w2, s2)
        fn = finpu.fp8_per_tensor_scale_moe
        kwargs.update(
            n_group=1, topk_group=1, routed_scaling_factor=1.0,
            use_routing_scales_on_input=False,
        )
    assert fn(*(t.npu() for t in args), out, **kwargs) is None
    fn(*args, expected, backend="reference", **kwargs)
    actual = out.cpu().float()
    assert torch.isfinite(actual).all()
    # The second GEMM consumes a dynamically quantized FP8 activation; the
    # reference keeps that activation in FP32. Bound the total quantization
    # error, including values near zero where elementwise relative error grows.
    assert (actual - expected.float()).norm() / expected.float().norm() < 0.05


def test_fp8_token_clustering_preserves_bits(torch_a5):
    torch = torch_a5
    from flashinfer.cann.moe import _cluster_tokens

    bits = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
    x = bits.npu().view(torch.float8_e4m3fn)
    ids = torch.tensor([[2, 0], [0, 2], [2, 0], [0, 2]])
    sorted_x, inverse, groups, _, _ = _cluster_tokens(x, ids.npu(), 4)
    order = ids.flatten().argsort(stable=True)
    expected = bits.repeat_interleave(2, 0)[order]
    assert groups == [4, 4, 8, 8]
    assert torch.equal(sorted_x.view(torch.uint8).cpu(), expected)
    restored = sorted_x.view(torch.uint8)[inverse].cpu()
    assert torch.equal(restored, bits.repeat_interleave(2, 0))
