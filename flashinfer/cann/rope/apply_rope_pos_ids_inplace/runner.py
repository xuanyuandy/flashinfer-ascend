"""Execution adapter for rope.apply_rope_pos_ids_inplace (cann)."""

from __future__ import annotations

from flashinfer.cann.rope.apply_rope_inplace.runner import _apply_rope_pos_ids_inplace_cann

run = _apply_rope_pos_ids_inplace_cann
