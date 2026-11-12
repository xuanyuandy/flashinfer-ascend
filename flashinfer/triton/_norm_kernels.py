"""Lazy compatibility alias for the operator-local device implementation."""

def __getattr__(name):
    from importlib import import_module
    return getattr(import_module("flashinfer.triton.norm.gemma_fused_add_rmsnorm.kernels"), name)
