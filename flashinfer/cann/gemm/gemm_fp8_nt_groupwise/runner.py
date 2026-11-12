"""Execution adapter for gemm.gemm_fp8_nt_groupwise (cann)."""

from __future__ import annotations
from flashinfer.cann.gemm._shared import (Any, _fp8_grouped_gemm)

def _gemm_fp8_nt_groupwise_cann(
    a: Any,
    b: Any,
    a_scale: Any,
    b_scale: Any,
    scale_major_mode: str,
    scale_granularity_mnk: Any,
    out: Any,
) -> Any:
    """gemm_fp8_nt_groupwise CANN 路径（950）：K 方向按 gk 分块做
    fp8 GEMM，分组 scale 按块折算累加（fp32 累加器）。"""
    import torch

    gm, gn, gk = (int(x) for x in scale_granularity_mnk)
    m, k = a.shape
    n = b.shape[0]
    sa, sb = a_scale.float(), b_scale.float()
    if sa.shape != (m // gm, k // gk):
        sa = sa.t()  # MN-major 存放 → 转置归一（同 reference）
    if sb.shape != (n // gn, k // gk):
        sb = sb.t()
    del scale_major_mode  # 布局差异已在转置归一中吸收（同 reference）
    y = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    for c in range(k // gk):
        sl = slice(c * gk, (c + 1) * gk)
        part = _fp8_grouped_gemm(
            a[:, sl].contiguous(),
            [b[:, sl].t().contiguous()],  # (gk, n)
            [m],
        ).float()  # (m, n)
        scale = sa[:, c].repeat_interleave(gm).unsqueeze(1) * sb[
            :, c
        ].repeat_interleave(gn).unsqueeze(0)
        y += part * scale
    out.copy_(y.to(out.dtype))
    return out

run = _gemm_fp8_nt_groupwise_cann
