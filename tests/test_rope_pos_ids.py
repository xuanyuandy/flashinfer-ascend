"""T2.3：ragged 布局 RoPE 五个新接口的契约与数值测试。

覆盖 ``apply_rope`` / ``apply_rope_inplace`` / ``apply_rope_pos_ids`` /
``apply_rope_pos_ids_inplace`` / ``apply_rope_with_cos_sin_cache``。

数值口径：CANN 实现与 float64 参考实现对拍。频率公式取自上游
``pos_enc.cuh`` 的 plain-rope 分支（smooth=0）：
``freq[j] = rope_theta ** (-2j / rotary_dim) / rope_scale``。
"""

import pytest

torch = pytest.importorskip("torch")

import flashinfer
from flashinfer.runtime import current_hardware

NPU = current_hardware().is_npu_available
requires_npu = pytest.mark.skipif(not NPU, reason="需要 NPU")
DEV = "npu" if NPU else "cpu"


def _qk(nnz=7, heads=4, head_dim=64, dtype=torch.float16):
    g = torch.Generator().manual_seed(1234)
    q = torch.randn(nnz, heads, head_dim, generator=g, dtype=torch.float32)
    k = torch.randn(nnz, heads, head_dim, generator=g, dtype=torch.float32)
    return q.to(dtype).to(DEV), k.to(dtype).to(DEV)


def _pos(nnz=7):
    return torch.arange(nnz, dtype=torch.int32, device=DEV)


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


# --------------------------------------------------------------- 契约

def test_upstream_signatures_match():
    """五个接口的位置参数名与顺序必须与上游逐字一致。"""
    import inspect

    expected = {
        "apply_rope": ["q", "k", "indptr", "offsets", "rotary_dim", "interleave",
                       "rope_scale", "rope_theta"],
        "apply_rope_inplace": ["q", "k", "indptr", "offsets", "rotary_dim",
                               "interleave", "rope_scale", "rope_theta"],
        "apply_rope_pos_ids": ["q", "k", "pos_ids", "rotary_dim", "interleave",
                               "rope_scale", "rope_theta"],
        "apply_rope_pos_ids_inplace": ["q", "k", "pos_ids", "rotary_dim",
                                       "interleave", "rope_scale", "rope_theta"],
        "apply_rope_with_cos_sin_cache": ["positions", "query", "key", "head_size",
                                          "cos_sin_cache", "is_neox"],
    }
    for name, want in expected.items():
        sig = inspect.signature(getattr(flashinfer, name))
        got = [p for p, v in sig.parameters.items()
               if v.kind is not inspect.Parameter.KEYWORD_ONLY]
        assert got == want, "%s 参数不匹配: %s != %s" % (name, got, want)


def test_defaults_match_upstream():
    """默认值与上游一致：rotary_dim=None, interleave=False, scale=1, theta=1e4。"""
    import inspect

    for name in ("apply_rope", "apply_rope_inplace",
                 "apply_rope_pos_ids", "apply_rope_pos_ids_inplace"):
        d = inspect.signature(getattr(flashinfer, name)).parameters
        assert d["rotary_dim"].default is None
        assert d["interleave"].default is False
        assert d["rope_scale"].default == 1
        assert d["rope_theta"].default == 1e4
    d = inspect.signature(flashinfer.apply_rope_with_cos_sin_cache).parameters
    assert d["is_neox"].default is True


def test_all_five_are_exported():
    for n in ("apply_rope", "apply_rope_inplace", "apply_rope_pos_ids",
              "apply_rope_pos_ids_inplace", "apply_rope_with_cos_sin_cache"):
        assert n in flashinfer.__all__, n


def test_every_op_has_cann_and_reference_kernel():
    from flashinfer.runtime.registry import get_global_registry

    ds = get_global_registry().descriptors()
    for op in ("rope.apply_rope", "rope.apply_rope_inplace",
               "rope.apply_rope_pos_ids", "rope.apply_rope_pos_ids_inplace",
               "rope.apply_rope_with_cos_sin_cache"):
        backends = {d.backend.value for d in ds if d.op == op}
        assert {"cann", "reference"} <= backends, "%s 只有 %s" % (op, backends)


def test_indptr_expansion_matches_segment_semantics():
    """indptr/offsets 展开出的位置必须等于 offsets[i] + 段内序号。"""
    from flashinfer.rope import _pos_ids_from_indptr

    indptr = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    offsets = torch.tensor([10, 100, 0], dtype=torch.int32)
    got = _pos_ids_from_indptr(indptr, offsets).tolist()
    assert got == [10, 11, 12, 100, 101, 0, 1, 2, 3]


# --------------------------------------------------------------- 数值

@requires_npu
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pos_ids_cann_matches_reference(interleave, dtype):
    q, k = _qk(dtype=dtype)
    pos = _pos()
    qr = flashinfer.apply_rope_pos_ids(
        q, k, pos, interleave=interleave, backend="reference")
    qc = flashinfer.apply_rope_pos_ids(
        q, k, pos, interleave=interleave, backend="cann")
    assert _rel_err(qc[0], qr[0]) < 2e-2
    assert _rel_err(qc[1], qr[1]) < 2e-2


@requires_npu
@pytest.mark.parametrize("rotary_dim", [None, 32])
def test_partial_rotary_leaves_tail_untouched(rotary_dim):
    """只旋转前 rotary_dim 维，其余维必须逐位不变。"""
    q, k = _qk()
    pos = _pos()
    out_q, _ = flashinfer.apply_rope_pos_ids(
        q, k, pos, rotary_dim=rotary_dim, backend="cann")
    rd = rotary_dim or q.shape[-1]
    if rd < q.shape[-1]:
        assert torch.equal(out_q[..., rd:], q[..., rd:])


@requires_npu
def test_inplace_modifies_input_and_returns_none():
    q, k = _qk()
    pos = _pos()
    q_before = q.clone()
    ret = flashinfer.apply_rope_pos_ids_inplace(q, k, pos, backend="cann")
    assert ret is None
    assert not torch.equal(q, q_before), "原地接口未修改输入"


@requires_npu
def test_out_of_place_does_not_modify_input():
    q, k = _qk()
    pos = _pos()
    q_before, k_before = q.clone(), k.clone()
    flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)


@requires_npu
def test_inplace_and_out_of_place_agree():
    q, k = _qk()
    pos = _pos()
    out_q, out_k = flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    qi, ki = q.clone(), k.clone()
    flashinfer.apply_rope_pos_ids_inplace(qi, ki, pos, backend="cann")
    assert torch.equal(qi, out_q)
    assert torch.equal(ki, out_k)


@requires_npu
def test_indptr_variant_matches_pos_ids_variant():
    """apply_rope 与等价的 apply_rope_pos_ids 必须给出相同结果。"""
    q, k = _qk(nnz=9)
    indptr = torch.tensor([0, 3, 5, 9], dtype=torch.int32, device=DEV)
    offsets = torch.tensor([10, 100, 0], dtype=torch.int32, device=DEV)
    a_q, a_k = flashinfer.apply_rope(q, k, indptr, offsets, backend="cann")
    pos = torch.tensor([10, 11, 12, 100, 101, 0, 1, 2, 3],
                       dtype=torch.int32, device=DEV)
    b_q, b_k = flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    assert torch.equal(a_q, b_q)
    assert torch.equal(a_k, b_k)


@requires_npu
@pytest.mark.parametrize("rope_scale,rope_theta", [(1, 1e4), (4, 1e4), (1, 5e5)])
def test_scale_and_theta_reach_the_kernel(rope_scale, rope_theta):
    """改变 scale/theta 必须改变结果，且与参考实现一致。"""
    q, k = _qk()
    pos = _pos()
    base, _ = flashinfer.apply_rope_pos_ids(q, k, pos, backend="cann")
    got, _ = flashinfer.apply_rope_pos_ids(
        q, k, pos, rope_scale=rope_scale, rope_theta=rope_theta, backend="cann")
    ref, _ = flashinfer.apply_rope_pos_ids(
        q, k, pos, rope_scale=rope_scale, rope_theta=rope_theta,
        backend="reference")
    assert _rel_err(got, ref) < 2e-2
    if (rope_scale, rope_theta) != (1, 1e4):
        assert not torch.equal(got, base), "参数未传到 kernel"


@requires_npu
def test_cos_sin_cache_out_of_place_matches_inplace():
    nnz, heads, hs = 7, 4, 64
    q = torch.randn(nnz, heads * hs, dtype=torch.float16, device=DEV)
    k = torch.randn(nnz, heads * hs, dtype=torch.float16, device=DEV)
    pos = _pos(nnz).to(torch.int64)
    cache = torch.randn(128, hs, dtype=torch.float32, device=DEV)
    out_q, out_k = flashinfer.apply_rope_with_cos_sin_cache(
        pos, q, k, hs, cache, backend="cann")
    qi, ki = q.clone(), k.clone()
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        pos, qi, ki, hs, cache, backend="cann")
    assert torch.equal(qi, out_q)
    assert torch.equal(ki, out_k)


def test_cos_sin_cache_rejects_non_float32():
    q = torch.randn(4, 64, dtype=torch.float16)
    k = torch.randn(4, 64, dtype=torch.float16)
    pos = torch.arange(4, dtype=torch.int64)
    cache = torch.randn(32, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="float32"):
        flashinfer.apply_rope_with_cos_sin_cache(pos, q, k, 64, cache)
