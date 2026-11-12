"""Execution adapter for rope.apply_rope_with_cos_sin_cache (reference)."""

from __future__ import annotations
from flashinfer.reference.rope.apply_rope_with_cos_sin_cache_inplace.runner import reference_apply_rope_with_cos_sin_cache_inplace

def reference_apply_rope_with_cos_sin_cache(positions, query, key, head_size, cos_sin_cache, is_neox):
    """``apply_rope_with_cos_sin_cache`` 参考实现（非原地）。"""
    q_out, k_out = query.clone(), key.clone()
    reference_apply_rope_with_cos_sin_cache_inplace(
        positions, q_out, k_out, head_size, cos_sin_cache, is_neox
    )
    return q_out, k_out

run = reference_apply_rope_with_cos_sin_cache
