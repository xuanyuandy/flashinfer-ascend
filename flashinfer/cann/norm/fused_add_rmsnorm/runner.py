"""Execution adapter for norm.fused_add_rmsnorm (cann)."""

from __future__ import annotations
from flashinfer.cann.norm._shared import (Any)

def _fused_add_rmsnorm_aclnn(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    import torch_npu

    y, _rstd, new_residual = torch_npu.npu_add_rms_norm(
        input, residual, weight, epsilon=eps
    )
    input.copy_(y)
    residual.copy_(new_residual)

run = _fused_add_rmsnorm_aclnn
