"""Execution adapter for attention.batch_prefill_ragged (cann)."""

from __future__ import annotations

from flashinfer.cann.attention.batch_decode_paged.runner import _batch_run

run = _batch_run
