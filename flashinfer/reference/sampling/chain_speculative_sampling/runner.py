"""Execution adapter for sampling.chain_speculative_sampling (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.sampling._family import execute

run = partial(execute, 'chain_speculative_sampling')
