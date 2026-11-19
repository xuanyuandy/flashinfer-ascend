"""Execution adapter for gemm.grouped_matmul_k_dim (reference)."""

from __future__ import annotations

from flashinfer.reference._shared import _require_torch


def reference_grouped_matmul_k_dim(x, weight, group_list, out, trans_a=False, fp32_output=False):
    """float64 reference: group g multiplies K rows [prev, prefix_g) of both operands.

    The accumulation runs on CPU in float64. Ascend has no double dtype and casts
    such operands down to float32, which would turn this oracle into a second
    approximate implementation and hide kernel error behind shared rounding.
    """
    torch = _require_torch()
    device = x.device
    x, weight = x.detach().cpu(), weight.detach().cpu()
    prefixes = group_list.to(torch.int64).cpu().tolist()
    total_k, n = int(weight.shape[0]), int(weight.shape[1])
    m = int(x.shape[1]) if trans_a else int(x.shape[0])
    dtype = torch.float32 if fp32_output else x.dtype
    y = torch.zeros((len(prefixes), m, n), dtype=dtype)
    start = 0
    for group, end in enumerate(prefixes):
        end = int(end)
        if end > start:
            left = x[start:end].double().T if trans_a else x[:, start:end].double()
            y[group] = (left @ weight[start:end].double()).to(dtype)
        start = end
    if start != total_k:
        raise ValueError("group_list must end at the K extent of weight")
    y = y.to(device)
    if out is None:
        return y
    out.copy_(y)
    return out


run = reference_grouped_matmul_k_dim
