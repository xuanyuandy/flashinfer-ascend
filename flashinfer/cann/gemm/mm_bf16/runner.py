"""Execution adapter for gemm.mm_bf16 (cann)."""

from __future__ import annotations

def _mm_bf16_cann(a, b, bias, out, out_dtype):
    """``(m,k) @ (k,n) -> (m,n)``；bias 形状 (n,)，按上游语义广播相加。"""
    import torch

    y = torch.matmul(a, b)
    if bias is not None:
        y = y + bias
    y = y.to(out_dtype)
    if out is None:
        return y
    out.copy_(y)
    return out

run = _mm_bf16_cann
