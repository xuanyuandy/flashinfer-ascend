"""Execution adapter for rope.apply_rope_pos_ids (cann)."""

from __future__ import annotations
from flashinfer.cann.rope._shared import (_apply_rope_pos_ids_cann)

def _apply_rope_pos_ids_out_cann(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta):
    return _apply_rope_pos_ids_cann(q, k, pos_ids, rotary_dim, interleave, rope_scale, rope_theta, False)

run = _apply_rope_pos_ids_out_cann
