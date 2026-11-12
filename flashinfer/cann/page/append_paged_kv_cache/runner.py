"""Execution adapter for page.append_paged_kv_cache (cann)."""

from __future__ import annotations

from functools import partial
from flashinfer.cann.page._family import append_execute

run = partial(append_execute)
