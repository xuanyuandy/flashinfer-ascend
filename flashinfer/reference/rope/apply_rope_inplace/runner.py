"""Execution adapter for rope.apply_rope_inplace (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_reference_rope_core)

def reference_apply_rope_pos_ids_inplace(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta):
    """``apply_rope_pos_ids_inplace`` 参考实现。"""
    return _reference_rope_core(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta, True)

run = reference_apply_rope_pos_ids_inplace
