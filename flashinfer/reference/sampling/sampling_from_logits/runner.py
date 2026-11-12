"""Execution adapter for sampling.sampling_from_logits (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.sampling._family import execute

run = partial(execute, 'sampling_from_logits')
