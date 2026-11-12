"""Execution adapter for norm.layernorm (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_layernorm(
    out: Any, input: Any, weight: Any, bias: Any, eps: float
) -> None:
    """LayerNorm 高精度参考实现（float64 累计）。

    ``out = (input - E[input]) / sqrt(Var[input] + eps) * weight + bias``，
    最后一维为归一化维。对标上游 flashinfer.norm.layernorm(v0.6.0)，
    其 gemma/beta 即此处 weight/bias。
    """
    torch = _require_torch()
    assert input.shape == out.shape, "layernorm: out 与 input 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "layernorm: weight 形状须为 (hidden_size,)"
    )
    assert bias.shape == weight.shape, "layernorm: bias 与 weight 形状须一致"
    with torch.no_grad():
        x = input.double()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + eps) * weight.double() + bias.double()
        out.copy_(y.to(out.dtype))

run = reference_layernorm
