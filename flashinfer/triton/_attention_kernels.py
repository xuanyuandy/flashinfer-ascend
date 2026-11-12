"""Lazy compatibility alias for the operator-local device implementation."""

def __getattr__(name):
    from importlib import import_module
    return getattr(import_module("flashinfer.triton.attention.single_prefill_catlass.kernels"), name)
