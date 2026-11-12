"""Execution adapter for sampling.top_p_sampling_from_probs (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.sampling._family import execute

run = partial(execute, 'top_p_sampling_from_probs')
