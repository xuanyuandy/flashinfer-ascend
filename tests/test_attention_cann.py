import math

import pytest
import torch

import flashinfer
from flashinfer._single_attention import single_attention
from flashinfer.attention.schema import AttentionMode
from flashinfer.attention import AttentionStateError
from flashinfer.cann.attention import dense_attention
from flashinfer.runtime import SchemaError


@pytest.fixture(scope="module")
def npu():
    import torch_npu  # noqa: F401
    assert torch.npu.is_available(), "public operator UT requires an available NPU"
    assert "950" in torch.npu.get_device_name(torch.npu.current_device()), "requires verified Ascend950"
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_single_fia_gqa_lse_and_layout(npu, dtype, decode, layout, head_dim):
    torch.manual_seed(19)
    q = torch.randn((4, head_dim) if decode else (3, 4, head_dim), dtype=dtype)
    k = torch.randn(7, 2, head_dim, dtype=dtype)
    v = torch.randn(7, 2, head_dim, dtype=dtype)
    if layout == "HND":
        k, v = k.transpose(0, 1), v.transpose(0, 1)
    expected, expected_lse = dense_attention(q, k, v, native=False, decode=decode, kv_layout=layout, causal=not decode, return_lse=True)
    call = flashinfer.single_decode_with_kv_cache if decode else flashinfer.single_prefill_with_kv_cache
    options = {} if decode else {"causal": True}
    actual, lse = call(q.to(npu), k.to(npu), v.to(npu), kv_layout=layout, return_lse=True, **options)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.015)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0.015, atol=0.015)


def test_zero_query_lse_base2_and_packed_mask(npu):
    q = torch.zeros(1, 2, 64, dtype=torch.float16, device=npu)
    k = torch.zeros(3, 1, 64, dtype=torch.float16, device=npu)
    v = torch.arange(3, dtype=torch.float16, device=npu)[:, None, None].expand(3, 1, 64).contiguous()
    out, lse = flashinfer.single_prefill_with_kv_cache(q, k, v, return_lse=True)
    torch.testing.assert_close(out.cpu(), torch.ones_like(out.cpu()), rtol=0, atol=0)
    torch.testing.assert_close(lse.cpu(), torch.full_like(lse.cpu(), math.log2(3)), rtol=1e-4, atol=1e-4)
    masked = flashinfer.single_prefill_with_kv_cache(q, k, v, packed_custom_mask=torch.tensor([2], dtype=torch.uint8, device=npu))
    torch.testing.assert_close(masked.cpu(), torch.ones_like(masked.cpu()), rtol=0, atol=0)
    q2 = q.expand(2, 2, 64).contiguous()
    mask = torch.tensor([[False, False, True], [False, False, True]], device=npu)
    custom = flashinfer.single_prefill_with_kv_cache(q2, k, v, custom_mask=mask, causal=True)
    torch.testing.assert_close(custom.cpu(), torch.full_like(custom.cpu(), 2), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("ragged", [False, True])
def test_batch_prefill_segments_pages_outputs_and_lse(npu, layout, ragged):
    torch.manual_seed(20)
    dtype = torch.float16
    q = torch.randn(3, 4, 64, dtype=dtype)
    keys = torch.randn(3, 2, 2, 64, dtype=dtype)
    values = torch.randn_like(keys)
    dense_k = [keys[2, :1], keys[0:2].reshape(4, 2, 64)[:3]]
    dense_v = [values[2, :1], values[0:2].reshape(4, 2, 64)[:3]]
    expected = [dense_attention(q[:1], dense_k[0], dense_v[0], native=False, causal=True, return_lse=True),
                dense_attention(q[1:], dense_k[1], dense_v[1], native=False, causal=True, return_lse=True)]
    cls = flashinfer.BatchPrefillWithRaggedKVCacheWrapper if ragged else flashinfer.BatchPrefillWithPagedKVCacheWrapper
    wrapper = cls(torch.empty(1024, dtype=torch.uint8, device=npu), kv_layout=layout)
    if ragged:
        plan_args = (torch.tensor([0, 1, 3], dtype=torch.int32), torch.tensor([0, 1, 4], dtype=torch.int32), 4, 2, 64)
        k, v = torch.cat(dense_k), torch.cat(dense_v)
        if layout == "HND":
            k, v = k.transpose(0, 1), v.transpose(0, 1)
        args = (q.to(npu), k.to(npu), v.to(npu))
    else:
        plan_args = (torch.tensor([0, 1, 3], dtype=torch.int32), torch.tensor([0, 1, 3], dtype=torch.int32),
                     torch.tensor([2, 0, 1], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32), 4, 2, 64, 2)
        if layout == "HND":
            keys, values = keys.transpose(1, 2), values.transpose(1, 2)
        args = (q.to(npu), (keys.to(npu), values.to(npu)))
    with pytest.raises(AttentionStateError, match="has not been planned"):
        wrapper.run(*args)
    wrapper.plan(*plan_args, causal=True)
    plan = wrapper.plan_state
    out = torch.empty(3, 4, 64, dtype=dtype, device=npu)
    lse = torch.empty(4, 3, dtype=torch.float32, device=npu)
    actual, actual_lse = wrapper.run(*args, out=out, lse=lse, return_lse=True, lse_base="ln", lse_layout="HN", enable_pdl=True)
    assert actual is out and actual_lse is lse
    torch.testing.assert_close(out.cpu(), torch.cat([e[0] for e in expected]), rtol=0.01, atol=0.01)
    torch.testing.assert_close(lse.cpu(), torch.cat([e[1] for e in expected]).t() * math.log(2), rtol=0.01, atol=0.01)
    assert wrapper.resolved_backend == "cann"
    assert wrapper.plan_selection.route == "builtin"
    assert wrapper.plan_selection.kernel_id.startswith("cann/")
    # Match upstream's stateful public lifecycle: plan once, run fresh data,
    # then replan on the same wrapper. A saved output is not a valid run cache.
    next_q = q + 0.125
    next_expected = torch.cat([
        dense_attention(next_q[:1], dense_k[0], dense_v[0], native=False, causal=True),
        dense_attention(next_q[1:], dense_k[1], dense_v[1], native=False, causal=True),
    ])
    next_args = (next_q.to(npu), *args[1:])
    torch.testing.assert_close(wrapper.run(*next_args).cpu(), next_expected, rtol=0.01, atol=0.01)
    assert wrapper.plan_state is plan
    wrapper.plan(*plan_args, causal=True)
    assert wrapper.plan_state.generation == plan.generation + 1
    torch.testing.assert_close(wrapper.run(*next_args).cpu(), next_expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_plan_metadata_and_workspace_reset(npu, layout, q_len, dtype):
    torch.manual_seed(23)
    q = torch.randn(2 * q_len, 4, 64, dtype=dtype)
    keys = torch.randn(3, 4, 2, 64, dtype=dtype)
    vals = torch.randn_like(keys)
    gathered_k = (keys[2, :3], keys[:2].reshape(8, 2, 64)[:5])
    gathered_v = (vals[2, :3], vals[:2].reshape(8, 2, 64)[:5])
    expected = [dense_attention(q[r * q_len:(r + 1) * q_len], gathered_k[r], gathered_v[r],
                                native=False, causal=q_len > 1, return_lse=True) for r in range(2)]
    ws = torch.empty(1024, dtype=torch.uint8, device=npu)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, layout, use_tensor_cores=q_len > 1)
    with pytest.raises(AttentionStateError, match="has not been planned"):
        wrapper.run(q.to(npu), (keys.to(npu), vals.to(npu)))
    wrapper.plan(torch.tensor([0, 1, 3], dtype=torch.int32), torch.tensor([2, 0, 1], dtype=torch.int32),
                 torch.tensor([3, 1], dtype=torch.int32), 4, 2, 64, 4, q_data_type=dtype, q_len_per_req=q_len)
    plan = wrapper.plan_state
    wrapper.reset_workspace_buffer(torch.empty_like(ws), torch.empty(0, dtype=torch.uint8, device=npu))
    assert wrapper.plan_state is plan
    if layout == "HND":
        keys, vals = keys.transpose(1, 2), vals.transpose(1, 2)
    output, lse = wrapper.run(q.to(npu), (keys.to(npu), vals.to(npu)), return_lse=True, q_len_per_req=q_len)
    torch.testing.assert_close(output.cpu(), torch.cat([x[0] for x in expected]), rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse.cpu(), torch.cat([x[1] for x in expected]), rtol=0.02, atol=0.02)
    next_q = q + 0.125
    next_expected = torch.cat([
        dense_attention(next_q[r * q_len:(r + 1) * q_len], gathered_k[r], gathered_v[r],
                        native=False, causal=q_len > 1) for r in range(2)
    ])
    actual = wrapper.run(next_q.to(npu), (keys.to(npu), vals.to(npu)), q_len_per_req=q_len)
    torch.testing.assert_close(actual.cpu(), next_expected, rtol=0.02, atol=0.02)
    assert wrapper.plan_state is plan
    with pytest.raises(SchemaError, match="q_len_per_req"):
        wrapper.run(q.to(npu), (keys.to(npu), vals.to(npu)), q_len_per_req=q_len + 1)
