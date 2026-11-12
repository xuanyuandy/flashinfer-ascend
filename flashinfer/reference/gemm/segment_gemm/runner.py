"""Execution adapter for gemm.segment_gemm (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_require_torch)

def reference_segment_gemm(x, weights, seg_indptr, weight_indices, weight_column_major, out):
    """分段 GEMM float64 参考实现：逐段取权重做矩阵乘。"""
    torch = _require_torch()
    idx = seg_indptr.to(torch.int64).tolist()
    n_seg = len(idx) - 1
    with torch.no_grad():
        parts = []
        for i in range(n_seg):
            w_i = int(weight_indices[i]) if weight_indices is not None else i
            w = weights[w_i].double()
            if weight_column_major:
                w = w.transpose(-1, -2)
            parts.append(torch.matmul(x[idx[i]:idx[i + 1]].double(), w))
        y = torch.cat(parts, dim=0).to(x.dtype)
    if out is None:
        return y
    out.copy_(y)
    return out

run = reference_segment_gemm
