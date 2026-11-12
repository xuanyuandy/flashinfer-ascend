"""Execution adapter for activation.gelu_and_mul (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.activation._family import execute

run = partial(execute, 'gelu_and_mul')
