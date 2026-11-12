"""Execution adapter for attention.batch_decode_paged (cann)."""

from __future__ import annotations
from flashinfer.cann.attention._shared import (SchemaError)

def _batch_run(wrapper, q, *kv, **options):
    from flashinfer.runtime._graph_params import _is_recording

    if _is_recording():
        raise SchemaError("batch FIA composite: graph_capability=not_capturable")
    return wrapper._run_native(q, *kv, **options)

run = _batch_run
