"""Execution adapter for sampling.top_p_renorm_probs (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.sampling._family import execute

run = partial(execute, 'top_p_renorm_probs')
