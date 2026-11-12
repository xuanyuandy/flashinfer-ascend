"""Execution adapter for page.append_paged_kv_cache (reference)."""

from __future__ import annotations

from functools import partial
from flashinfer.reference.page._family import append_execute

run = partial(append_execute)
