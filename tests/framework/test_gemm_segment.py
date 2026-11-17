"""T2.4：mm_bf16 / bmm_bf16 / SegmentGEMMWrapper 的契约与数值测试。

数值口径：CANN 实现与 float64 参考实现对拍。
"""

import pytest

torch = pytest.importorskip("torch")

import flashinfer
from flashinfer.runtime import current_hardware

NPU = current_hardware().is_npu_available
requires_npu = pytest.mark.skipif(not NPU, reason="需要 NPU")
DEV = "npu" if NPU else "cpu"


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.float().norm().clamp_min(1e-12)).item()


def _rand(*shape, dtype=torch.bfloat16, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype).to(DEV)


# --------------------------------------------------------------- 契约

def test_signatures_match_upstream():
    import inspect

    exp = {
        "mm_bf16": ["a", "b", "bias", "pdl", "out", "out_dtype", "backend"],
        "bmm_bf16": ["A", "B", "out", "out_dtype", "backend"],
    }
    for name, want in exp.items():
        got = [p for p, v in inspect.signature(getattr(flashinfer, name)).parameters.items()
               if v.kind is not inspect.Parameter.KEYWORD_ONLY]
        assert got == want, "%s: %s != %s" % (name, got, want)

    run = inspect.signature(flashinfer.SegmentGEMMWrapper.run).parameters
    got = [p for p, v in run.items()
           if p != "self" and v.kind is not inspect.Parameter.KEYWORD_ONLY]
    assert got == ["x", "weights", "batch_size", "weight_column_major",
                   "out", "seg_lens", "seg_indptr", "weight_indices"]


def test_wrapper_surface_matches_upstream():
    for m in ("run", "reset_workspace_buffer"):
        assert hasattr(flashinfer.SegmentGEMMWrapper, m), m
    flashinfer.SegmentGEMMWrapper(None).reset_workspace_buffer(None, None)


def test_exports():
    for n in ("mm_bf16", "bmm_bf16", "SegmentGEMMWrapper"):
        assert n in flashinfer.__all__, n


def test_every_op_has_cann_and_reference():
    from flashinfer.runtime.registry import get_global_registry

    ds = get_global_registry().descriptors()
    for op in ("gemm.mm_bf16", "gemm.bmm_bf16", "gemm.segment_gemm"):
        b = {d.backend.value for d in ds if d.op == op}
        assert {"cann", "reference"} <= b, "%s 只有 %s" % (op, b)


def test_segment_gemm_requires_one_of_seg_lens_or_indptr():
    w = flashinfer.SegmentGEMMWrapper(None)
    with pytest.raises(ValueError, match="seg_lens"):
        w.run(torch.zeros(2, 2), torch.zeros(1, 2, 2), 1, False)


# --------------------------------------------------------------- 数值

@requires_npu
@pytest.mark.parametrize("bias", [False, True])
def test_mm_bf16_matches_reference(bias):
    a, b = _rand(32, 64), _rand(64, 48, seed=8)
    bi = _rand(48, seed=9) if bias else None
    got = flashinfer.mm_bf16(a, b, bi, backend="cann")
    ref = flashinfer.mm_bf16(a, b, bi, backend="reference")
    assert got.shape == (32, 48)
    assert _rel(got, ref) < 2e-2


@requires_npu
def test_bmm_bf16_matches_reference():
    A, B = _rand(5, 16, 32), _rand(5, 32, 24, seed=8)
    got = flashinfer.bmm_bf16(A, B, backend="cann")
    ref = flashinfer.bmm_bf16(A, B, backend="reference")
    assert got.shape == (5, 16, 24)
    assert _rel(got, ref) < 2e-2


@requires_npu
def test_mm_bf16_accepts_and_ignores_pdl():
    """pdl 是 CUDA 专属；接受但不得改变结果，也不得报错。"""
    a, b = _rand(16, 32), _rand(32, 16, seed=8)
    off = flashinfer.mm_bf16(a, b, pdl=False, backend="cann")
    on = flashinfer.mm_bf16(a, b, pdl=True, backend="cann")
    assert torch.equal(off, on)


@requires_npu
def test_mm_bf16_writes_into_out():
    a, b = _rand(8, 16), _rand(16, 12, seed=8)
    out = torch.empty(8, 12, dtype=torch.bfloat16, device=DEV)
    ret = flashinfer.mm_bf16(a, b, out=out, backend="cann")
    assert ret.data_ptr() == out.data_ptr()
    assert _rel(out, flashinfer.mm_bf16(a, b, backend="reference")) < 2e-2


@requires_npu
@pytest.mark.parametrize("column_major", [False, True])
def test_segment_gemm_matches_reference(column_major):
    seg_lens = torch.tensor([3, 5, 2], dtype=torch.int64, device=DEV)
    d_in, d_out = 32, 16
    x = _rand(int(seg_lens.sum()), d_in)
    w = _rand(3, d_out, d_in, seed=8) if column_major else _rand(3, d_in, d_out, seed=8)
    wr = flashinfer.SegmentGEMMWrapper(None)
    got = wr.run(x, w, 3, column_major, seg_lens=seg_lens, backend="cann")
    ref = wr.run(x, w, 3, column_major, seg_lens=seg_lens, backend="reference")
    assert got.shape == (int(seg_lens.sum()), d_out)
    assert _rel(got, ref) < 2e-2
    # Upstream SegmentGEMMWrapper has no plan(): the segment metadata belongs
    # to each run. Reuse the wrapper with fresh input and explicit output.
    next_x = x + 0.125
    out = torch.empty_like(got)
    returned = wr.run(next_x, w, 3, column_major, seg_lens=seg_lens, out=out, backend="cann")
    expected = wr.run(next_x, w, 3, column_major, seg_lens=seg_lens, backend="reference")
    assert returned is out
    assert _rel(out, expected) < 2e-2


@requires_npu
def test_segment_gemm_weight_indices_select_per_segment():
    """weight_indices 必须真正改变每段使用的权重。"""
    seg_lens = torch.tensor([4, 4], dtype=torch.int64, device=DEV)
    x = _rand(8, 16)
    w = _rand(3, 16, 8, seed=8)
    wr = flashinfer.SegmentGEMMWrapper(None)
    a = wr.run(x, w, 2, False, seg_lens=seg_lens,
               weight_indices=torch.tensor([0, 1], device=DEV), backend="cann")
    b = wr.run(x, w, 2, False, seg_lens=seg_lens,
               weight_indices=torch.tensor([2, 0], device=DEV), backend="cann")
    ref = wr.run(x, w, 2, False, seg_lens=seg_lens,
                 weight_indices=torch.tensor([2, 0], device=DEV), backend="reference")
    assert not torch.equal(a, b), "weight_indices 未生效"
    assert _rel(b, ref) < 2e-2


@requires_npu
def test_segment_gemm_seg_lens_and_indptr_agree():
    seg_lens = torch.tensor([2, 3, 4], dtype=torch.int64, device=DEV)
    indptr = torch.tensor([0, 2, 5, 9], dtype=torch.int64, device=DEV)
    x = _rand(9, 16)
    w = _rand(3, 16, 8, seed=8)
    wr = flashinfer.SegmentGEMMWrapper(None)
    a = wr.run(x, w, 3, False, seg_lens=seg_lens, backend="cann")
    b = wr.run(x, w, 3, False, seg_indptr=indptr, backend="cann")
    assert torch.equal(a, b)
