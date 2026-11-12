"""Execution adapter for sampling.sampling_from_logits (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.sampling._family import execute

run = partial(execute, 'sampling_from_logits')
