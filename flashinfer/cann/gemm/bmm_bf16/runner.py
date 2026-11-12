"""Execution adapter for gemm.bmm_bf16 (cann)."""

from __future__ import annotations

def _bmm_bf16_cann(A, B, out, out_dtype):
    """``(b,m,k) @ (b,k,n) -> (b,m,n)``。"""
    import torch

    y = torch.bmm(A, B).to(out_dtype)
    if out is None:
        return y
    out.copy_(y)
    return out

run = _bmm_bf16_cann
