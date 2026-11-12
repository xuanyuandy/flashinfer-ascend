"""Execution adapter for norm.fused_add_rmsnorm (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_fused_add_rmsnorm(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    """Fused add RMSNorm 高精度参考实现（float64 累计，原地修改）。

    Step 1: ``residual[i] += input[i]``
    Step 2: ``input[i] = (residual[i] / RMS(residual)) * weight[i]``
    """
    torch = _require_torch()
    assert input.shape == residual.shape, "fused_add_rmsnorm: 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "fused_add_rmsnorm: weight 形状须为 (hidden_size,)"
    )
    with torch.no_grad():
        new_residual = input.double() + residual.double()
        residual.copy_(new_residual.to(residual.dtype))
        rstd = torch.rsqrt(new_residual.pow(2).mean(dim=-1, keepdim=True) + eps)
        input.copy_((new_residual * rstd * weight.double()).to(input.dtype))

run = reference_fused_add_rmsnorm
