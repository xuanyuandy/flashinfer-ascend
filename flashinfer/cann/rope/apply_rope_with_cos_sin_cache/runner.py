"""Execution adapter for rope.apply_rope_with_cos_sin_cache (cann)."""

from __future__ import annotations
from flashinfer.cann.rope.apply_rope_with_cos_sin_cache_inplace.runner import _apply_rope_cos_sin_cache_aclnn

def _apply_rope_cos_sin_cache_out_cann(positions, query, key, head_size, cos_sin_cache, is_neox):
    """``apply_rope_with_cos_sin_cache`` 的非原地版本：克隆后复用原地实现。"""
    q_out, k_out = query.clone(), key.clone()
    _apply_rope_cos_sin_cache_aclnn(positions, q_out, k_out, head_size, cos_sin_cache, is_neox)
    return q_out, k_out

run = _apply_rope_cos_sin_cache_out_cann
