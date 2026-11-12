"""Execution adapter for gemm.mm_bf16 (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_require_torch)

def reference_mm_bf16(a, b, bias, out, out_dtype):
    """``mm_bf16`` float64 参考实现。"""
    torch = _require_torch()
    with torch.no_grad():
        y = torch.matmul(a.double(), b.double())
        if bias is not None:
            y = y + bias.double()
    y = y.to(out_dtype)
    if out is None:
        return y
    out.copy_(y)
    return out

run = reference_mm_bf16
