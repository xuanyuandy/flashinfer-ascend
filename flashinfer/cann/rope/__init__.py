"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.rope._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _SUPPORTED_DTYPES, _SUPPORTED_SOCS, _apply_rope_pos_ids_cann, _rope_cos_sin, _rotate_3d)
from flashinfer.cann.rope.apply_rope_with_cos_sin_cache_inplace.runner import _apply_rope_cos_sin_cache_aclnn
from flashinfer.cann.rope.apply_rope_with_cos_sin_cache.runner import _apply_rope_cos_sin_cache_out_cann
from flashinfer.cann.rope.apply_rope_inplace.runner import _apply_rope_pos_ids_inplace_cann
from flashinfer.cann.rope.apply_rope_pos_ids.runner import _apply_rope_pos_ids_out_cann


def rope_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.rope')
