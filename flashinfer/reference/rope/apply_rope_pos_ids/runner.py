"""Execution adapter for rope.apply_rope_pos_ids (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_reference_rope_core)

def reference_apply_rope_pos_ids(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta):
    """``apply_rope_pos_ids`` 参考实现（非原地）。"""
    return _reference_rope_core(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta, False)

run = reference_apply_rope_pos_ids
