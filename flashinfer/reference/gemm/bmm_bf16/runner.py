"""Execution adapter for gemm.bmm_bf16 (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_require_torch)

def reference_bmm_bf16(A, B, out, out_dtype):
    """``bmm_bf16`` float64 参考实现。"""
    torch = _require_torch()
    with torch.no_grad():
        y = torch.bmm(A.double(), B.double()).to(out_dtype)
    if out is None:
        return y
    out.copy_(y)
    return out

run = reference_bmm_bf16
