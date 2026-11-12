"""Execution adapter for attention.single_prefill (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.attention._shared import dense_attention

run = partial(dense_attention, native=True)
