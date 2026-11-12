"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.cascade._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _SUPPORTED_DTYPES, _SUPPORTED_SOCS)
from flashinfer.cann.cascade.merge_state.runner import _merge_state_attention_update


def cascade_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.cascade')
