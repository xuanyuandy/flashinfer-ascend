"""Execution adapter for sampling.min_p_sampling_from_probs (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.sampling._family import execute

run = partial(execute, 'min_p_sampling_from_probs')
