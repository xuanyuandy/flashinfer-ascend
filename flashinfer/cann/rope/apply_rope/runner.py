"""Execution adapter for rope.apply_rope (cann)."""

from __future__ import annotations

from flashinfer.cann.rope.apply_rope_pos_ids.runner import _apply_rope_pos_ids_out_cann

run = _apply_rope_pos_ids_out_cann
