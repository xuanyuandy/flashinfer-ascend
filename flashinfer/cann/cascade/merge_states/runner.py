"""Execution adapter for cascade.merge_states (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.cascade._family import merge_states_execute

run = partial(merge_states_execute)
