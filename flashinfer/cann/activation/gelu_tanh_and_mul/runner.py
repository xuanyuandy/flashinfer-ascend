"""Execution adapter for activation.gelu_tanh_and_mul (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.activation._family import execute

run = partial(execute, 'gelu_tanh_and_mul')
