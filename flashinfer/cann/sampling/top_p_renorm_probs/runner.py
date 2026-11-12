"""Execution adapter for sampling.top_p_renorm_probs (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.sampling._family import execute

run = partial(execute, 'top_p_renorm_probs')
