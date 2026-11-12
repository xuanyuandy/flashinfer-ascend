"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.decode._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _SUPPORTED_DTYPES, _SUPPORTED_SOCS)
from flashinfer.cann.decode.cudnn_batch_decode_with_kv_cache.runner import _cudnn_batch_decode_fia


def decode_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.decode')
