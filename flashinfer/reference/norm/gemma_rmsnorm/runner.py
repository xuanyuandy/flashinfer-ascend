"""Execution adapter for norm.gemma_rmsnorm (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_gemma_rmsnorm(out: Any, input: Any, weight: Any, eps: float) -> None:
    """Gemma RMSNorm 高精度参考实现（float64 累计）。

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``——与 rmsnorm
    的唯一差异是缩放因子为 ``weight + 1``（Gemma 约定）。
    """
    torch = _require_torch()
    assert input.shape == out.shape, "gemma_rmsnorm: out 与 input 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "gemma_rmsnorm: weight 形状须为 (hidden_size,)"
    )
    with torch.no_grad():
        x = input.double()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        out.copy_((x * rstd * (weight.double() + 1.0)).to(out.dtype))

run = reference_gemma_rmsnorm
