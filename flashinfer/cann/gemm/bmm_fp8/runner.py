"""Execution adapter for gemm.bmm_fp8 (cann)."""

from __future__ import annotations
from flashinfer.cann.gemm._shared import (Any, _fp8_grouped_gemm)

def _bmm_fp8_cann(
    A: Any, B: Any, A_scale: Any, B_scale: Any, out: Any
) -> Any:
    """bmm_fp8 CANN 路径（950）：batch 作为分组走 fp8 grouped GEMM，
    per-tensor scale 在 GEMM 之后统一折算。"""
    b, m, k = A.shape
    n = B.shape[2]
    x = A.reshape(b * m, k).contiguous()
    weight_list = [B[i].contiguous() for i in range(b)]  # 每组 (k, n)
    group_list = [(i + 1) * m for i in range(b)]
    y = _fp8_grouped_gemm(x, weight_list, group_list).float().reshape(b, m, n)
    y = y * A_scale.float() * B_scale.float()
    out.copy_(y.to(out.dtype))
    return out

run = _bmm_fp8_cann
