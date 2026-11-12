"""Compatibility exports and domain implementation collection."""
from .gemma_fused_add_rmsnorm.runner import _gemma_fused_add_rmsnorm_triton

def norm_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend="triton", domain='norm')
