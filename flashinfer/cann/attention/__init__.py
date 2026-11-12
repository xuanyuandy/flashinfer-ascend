"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.attention._shared import Backend, CapabilitySpec, GraphCapability, GraphReplaySpec, KernelDescriptor, SchemaError, _DECODE_ID, _DECODE_REPLAY, _scale, dense_attention, maximum_workspace_bytes, partial
from flashinfer.cann.attention.batch_decode_paged.runner import _batch_run


def attention_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.attention')
