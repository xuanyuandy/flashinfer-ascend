"""Execution adapter for sampling.top_k_renorm_probs (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.sampling._family import execute

run = partial(execute, 'top_k_renorm_probs')
