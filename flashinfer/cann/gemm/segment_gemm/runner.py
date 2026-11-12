"""Execution adapter for gemm.segment_gemm (cann)."""

from __future__ import annotations

def _segment_gemm_cann(x, weights, seg_indptr, weight_indices, weight_column_major, out):
    """分段 GEMM：``y[i] = x[i] @ W[weight_indices[i]]``，走 npu_grouped_matmul。

    ``seg_indptr`` 为含前导 0 的累加和；group_list 取其尾部（每段结束位置），
    与 npu_grouped_matmul 的 group_type=0 语义一致。
    """
    import torch
    import torch_npu

    idx = seg_indptr.to(torch.int64)
    group_list = idx[1:].to(torch.int64)
    n_groups = int(group_list.numel())
    sel = (
        weight_indices.to(torch.int64)
        if weight_indices is not None
        else torch.arange(n_groups, device=x.device, dtype=torch.int64)
    )
    w = weights.index_select(0, sel)
    if weight_column_major:
        # 上游 column major 是 (num_weights, d_out, d_in)，转成 (num_weights, d_in, d_out)
        w = w.transpose(-1, -2)
    w = w.contiguous()
    y = torch_npu.npu_grouped_matmul(
        [x], [w], group_list=group_list, split_item=3, group_type=0
    )[0]
    if out is None:
        return y
    out.copy_(y)
    return out

run = _segment_gemm_cann
