"""Compatibility exports and domain implementation collection."""
from .matmul.runner import TritonMatmulRunner

def gemm_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend="triton", domain='gemm')
