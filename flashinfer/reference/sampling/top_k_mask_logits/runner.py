"""Execution adapter for sampling.top_k_mask_logits (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.sampling._family import execute

run = partial(execute, 'top_k_mask_logits')
