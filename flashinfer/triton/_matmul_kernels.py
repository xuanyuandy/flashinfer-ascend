"""Lazy compatibility alias for the operator-local device implementation."""

def __getattr__(name):
    from importlib import import_module
    return getattr(import_module("flashinfer.triton.gemm.matmul.kernels"), name)
