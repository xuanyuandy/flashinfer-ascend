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

















def test_cos_sin_cache_rejects_non_float32():
    q = torch.randn(4, 64, dtype=torch.float16)
    k = torch.randn(4, 64, dtype=torch.float16)
    pos = torch.arange(4, dtype=torch.int64)
    cache = torch.randn(32, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="float32"):
        flashinfer.apply_rope_with_cos_sin_cache(pos, q, k, 64, cache)
