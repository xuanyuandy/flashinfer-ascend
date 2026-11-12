"""Execution adapter for norm.rmsnorm (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_rmsnorm(out: Any, input: Any, weight: Any, eps: float) -> None:
    """RMSNorm 高精度参考实现（float64 累计）。

    ``out[i] = (input[i] / RMS(input)) * weight[i]``，最后一维为归一化维。
    """
    torch = _require_torch()
    assert input.shape == out.shape, "rmsnorm: out 与 input 形状须一致"
    assert weight.shape == input.shape[-1:], "rmsnorm: weight 形状须为 (hidden_size,)"
    with torch.no_grad():
        x = input.double()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        out.copy_((x * rstd * weight.double()).to(out.dtype))

run = reference_rmsnorm
