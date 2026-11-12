"""Execution adapter for gemm.bmm_fp8 (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_bmm_fp8(
    A: Any,
    B: Any,
    A_scale: Any,
    B_scale: Any,
    out: Any,
) -> Any:
    """fp8 per-tensor 批量矩阵乘参考实现（反量化到 float32 计算）。

    对标上游 ``flashinfer.gemm.bmm_fp8``：A (b, m, k)、B (b, k, n) 为
    fp8(e4m3/e5m2)，A_scale/B_scale 为 per-tensor 标量 scale，
    ``out = (A @ B) * A_scale * B_scale``，输出 bf16/fp16。
    """
    torch = _require_torch()
    with torch.no_grad():
        acc = torch.matmul(A.float(), B.float())
        acc = acc * A_scale.float() * B_scale.float()
        out.copy_(acc.to(out.dtype))
    return out

run = reference_bmm_fp8
