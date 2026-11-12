"""Execution adapter for norm.gemma_rmsnorm (cann)."""

from __future__ import annotations
from flashinfer.cann.norm._shared import (Any)

def _gemma_rmsnorm_aclnn(out: Any, input: Any, weight: Any, eps: float) -> None:
    import torch_npu

    y, _rstd = torch_npu.npu_gemma_rms_norm(input, weight, epsilon=eps)
    out.copy_(y)

run = _gemma_rmsnorm_aclnn
