"""Execution adapter for sampling.min_p_sampling_from_probs (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.sampling._family import execute

run = partial(execute, 'min_p_sampling_from_probs')
