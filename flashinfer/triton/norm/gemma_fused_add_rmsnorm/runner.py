from __future__ import annotations
from typing import Any, Tuple
from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")

def _gemma_fused_add_rmsnorm_triton(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    from flashinfer.triton.norm.gemma_fused_add_rmsnorm import kernels as _norm_kernels

    _norm_kernels.gemma_fused_add_rmsnorm(input, residual, weight, eps)
