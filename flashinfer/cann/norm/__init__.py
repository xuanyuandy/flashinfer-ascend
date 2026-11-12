"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.norm._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _SUPPORTED_DTYPES, _SUPPORTED_SOCS)
from flashinfer.cann.norm.fused_add_rmsnorm.runner import _fused_add_rmsnorm_aclnn
from flashinfer.cann.norm.gemma_rmsnorm.runner import _gemma_rmsnorm_aclnn
from flashinfer.cann.norm.layernorm.runner import _layernorm_aclnn
from flashinfer.cann.norm.rmsnorm.runner import _rmsnorm_aclnn


def norm_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.norm')
