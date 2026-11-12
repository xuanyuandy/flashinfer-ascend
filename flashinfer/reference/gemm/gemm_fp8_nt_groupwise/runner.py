"""Execution adapter for gemm.gemm_fp8_nt_groupwise (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_gemm_fp8_nt_groupwise(
    a: Any,
    b: Any,
    a_scale: Any,
    b_scale: Any,
    scale_major_mode: str,
    scale_granularity_mnk: Any,
    out: Any,
) -> Any:
    """fp8 分组 scale 矩阵乘参考实现（反量化到 float32 计算）。

    对标上游 ``flashinfer.gemm.gemm_fp8_nt_groupwise``：a (m, k) 行主序、
    b (n, k) 列主序（NT 布局），分组粒度 (gm, gn, gk)；
    scale_major_mode="K" 时 a_scale (m//gm, k//gk)、b_scale (n//gn, k//gk)；
    "MN" 时两 scale 转置存放。``out = (a ⊙ sa) @ (b ⊙ sb)^T``。
    """
    gm, gn, gk = (int(x) for x in scale_granularity_mnk)

    def _expand(scale, rows, cols, g_r, g_c):
        # scale 归一到 (rows//g_r, cols//g_c) 后按粒度扩展到 (rows, cols)
        s = scale.float()
        if s.shape != (rows // g_r, cols // g_c):
            s = s.t()  # MN-major 存放 → 转置回 (rows, cols) 分块视角
        assert s.shape == (rows // g_r, cols // g_c), (
            f"scale 形状 {tuple(s.shape)} 与粒度 {(g_r, g_c)} 不匹配"
        )
        return s.repeat_interleave(g_r, 0).repeat_interleave(g_c, 1)

    m, k = a.shape
    n = b.shape[0]
    sa = _expand(a_scale, m, k, gm, gk)
    sb = _expand(b_scale, n, k, gn, gk)
    del scale_major_mode  # 布局差异已在 _expand 转置归一中吸收
    torch = _require_torch()
    with torch.no_grad():
        acc = torch.matmul(a.float() * sa, (b.float() * sb).t())
        out.copy_(acc.to(out.dtype))
    return out

run = reference_gemm_fp8_nt_groupwise
