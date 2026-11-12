"""Execution adapter for cascade.merge_states (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.cascade._family import merge_states_execute

run = partial(merge_states_execute)
