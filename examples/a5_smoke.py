"""Run public FlashInfer APIs on A5 and check against explicit CPU references.

    python examples/a5_smoke.py
    python examples/a5_smoke.py --case gemm --device 0
"""

import argparse

import torch
import torch_npu

import flashinfer as fi


def check(name, actual, expected, *, relative_limit=None):
    if actual.device.type != "npu":
        raise AssertionError(f"{name}: expected NPU execution, got {actual.device}")
    actual = actual.cpu().float()
    expected = expected.float()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise AssertionError(f"{name}: invalid output")
    relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    if relative_limit is None:
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.01)
    elif relative >= relative_limit:
        raise AssertionError(f"{name}: relative error {relative.item():.4g}")
    print(f"PASS {name:32s} shape={tuple(actual.shape)} rel_error={relative.item():.3g}")


def norm():
    x = torch.randn(8, 4096, dtype=torch.bfloat16)
    w = torch.randn(4096, dtype=torch.bfloat16)
    b = torch.randn_like(w)
    for fn in (fi.rmsnorm, fi.gemma_rmsnorm, fi.layernorm):
        args = (x, w, b) if fn is fi.layernorm else (x, w)
        check(fn.__name__, fn(*(t.npu() for t in args)), fn(*args, backend="reference"))
    for fn in (fi.fused_add_rmsnorm, fi.gemma_fused_add_rmsnorm):
        residual = torch.randn_like(x)
        xn, rn = x.npu(), residual.npu()
        xc, rc = x.clone(), residual.clone()
        fn(xn, rn, w.npu())
        fn(xc, rc, w, backend="reference")
        check(fn.__name__, xn, xc)
        check(fn.__name__ + ".residual", rn, rc)


def rope():
    positions = torch.arange(8)
    cache = torch.randn(32, 64)
    for neox in (True, False):
        q = torch.randn(8, 4 * 128, dtype=torch.bfloat16)
        k = torch.randn(8, 2 * 128, dtype=torch.bfloat16)
        qn, kn = q.npu(), k.npu()
        fi.apply_rope_with_cos_sin_cache_inplace(
            positions.npu(), qn, kn, 128, cache.npu(), neox
        )
        fi.apply_rope_with_cos_sin_cache_inplace(
            positions, q, k, 128, cache, neox, backend="reference"
        )
        check(f"rope.neox={neox}.query", qn, q)
        check(f"rope.neox={neox}.key", kn, k)


def decode():
    q = torch.randn(3, 8, 64, dtype=torch.bfloat16)
    k = torch.randn(10, 2, 16, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([5, 17, 40], dtype=torch.int64)
    tables = torch.randperm(10)[:9].reshape(3, 3).int()
    kwargs = dict(max_sequence_kv=48, actual_seq_lens_kv=lengths)
    actual = fi.cudnn_batch_decode_with_kv_cache(
        q.npu(), k.npu(), v.npu(), 0.125, None, block_tables=tables.npu(), **kwargs
    )
    expected = fi.cudnn_batch_decode_with_kv_cache(
        q, k, v, 0.125, None, block_tables=tables, backend="reference", **kwargs
    )
    check("paged_decode", actual, expected)


def cascade():
    args = (
        torch.randn(8, 4, 64, dtype=torch.bfloat16), torch.randn(8, 4),
        torch.randn(8, 4, 64, dtype=torch.bfloat16), torch.randn(8, 4),
    )
    actual = fi.merge_state(*(t.npu() for t in args))
    expected = fi.merge_state(*args, backend="reference")
    for name, value, reference in zip(("output", "lse"), actual, expected):
        check("merge_state." + name, value, reference)


def gemm():
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        a = (torch.randn(2, 8, 128) * 10).to(dtype)
        b = (torch.randn(2, 128, 64) * 10).to(dtype)
        sa, sb = torch.tensor([0.01]), torch.tensor([0.02])
        actual = fi.bmm_fp8(a.npu(), b.npu(), sa.npu(), sb.npu(), torch.bfloat16)
        expected = fi.bmm_fp8(a, b, sa, sb, torch.bfloat16, backend="reference")
        check(f"bmm.{dtype}", actual, expected)
        a = (torch.randn(32, 256) * 10).to(dtype)
        b = (torch.randn(128, 256) * 10).to(dtype)
        for mode in ("K", "MN"):
            sa, sb = torch.rand(32, 2) * 0.02, torch.rand(1, 2) * 0.02
            if mode == "MN":
                sa, sb = sa.t().contiguous(), sb.t().contiguous()
            actual = fi.gemm_fp8_nt_groupwise(a.npu(), b.npu(), sa.npu(), sb.npu(), mode)
            expected = fi.gemm_fp8_nt_groupwise(a, b, sa, sb, mode, backend="reference")
            check(f"groupwise.{dtype}.{mode}", actual, expected)


def moe():
    seq, hidden, inter, experts = 8, 256, 256, 4
    logits = torch.randn(seq, experts)
    logits[:, 2:] = -100  # Exercise empty expert groups.
    bias = torch.zeros(experts)
    x = torch.randn(seq, hidden, dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16) * 0.05
    scales, ids = logits.softmax(-1).topk(2, dim=-1)
    args = (x, ids, scales, w1, w2)
    actual = fi.fused_moe(*(t.npu() for t in args), torch.bfloat16, [])
    expected = fi.fused_moe(*args, torch.bfloat16, [], backend="reference")
    check("moe.bfloat16", actual, expected, relative_limit=0.02)

    x, w1, w2 = (t.to(torch.float8_e4m3fn) for t in (x, w1, w2))
    common = dict(
        num_experts=experts, top_k=2, intermediate_size=inter,
        local_expert_offset=0, local_num_experts=experts, routing_method_type=0,
    )
    for block in (False, True):
        if block:
            args = (
                logits, bias, x, torch.rand(hidden // 128, seq) + 0.5,
                w1, torch.rand(experts, 2 * inter // 128, hidden // 128) + 0.5,
                w2, torch.rand(experts, hidden // 128, inter // 128) + 0.5,
            )
            fn = fi.fp8_block_scale_moe
            kwargs = dict(n_groups=1, top_k_groups=1, routed_scaling=1.0)
        else:
            args = (
                logits, bias, x, w1, torch.ones(experts), torch.ones(experts),
                w2, torch.ones(experts),
            )
            fn = fi.fp8_per_tensor_scale_moe
            kwargs = dict(
                n_group=1, topk_group=1, routed_scaling_factor=1.0,
                use_routing_scales_on_input=False,
            )
        actual = torch.empty(seq, hidden, dtype=torch.bfloat16, device="npu")
        expected = torch.empty(seq, hidden, dtype=torch.bfloat16)
        fn(*(t.npu() for t in args), actual, **common, **kwargs)
        fn(*args, expected, backend="reference", **common, **kwargs)
        check(fn.__name__, actual, expected, relative_limit=0.05)


@torch.inference_mode()
def main():
    cases = dict(norm=norm, rope=rope, decode=decode, cascade=cascade, gemm=gemm, moe=moe)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *cases], default="all")
    parser.add_argument("--device", type=int, default=0, help="visible NPU index")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    hardware = fi.current_hardware()
    if hardware.soc_model != "Ascend950":
        raise RuntimeError(f"This example requires A5 (Ascend950), found {hardware}")
    torch.manual_seed(0)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
    print(f"device={torch.npu.get_device_name(args.device)} hardware={hardware}")
    for name, run in cases.items():
        if args.case in ("all", name):
            run()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
