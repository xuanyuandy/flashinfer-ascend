"""Execution adapter for fused_moe.trtllm_fp8_block_scale_moe (cann)."""

from __future__ import annotations
from flashinfer.cann.fused_moe._shared import (Any, _aggregate_expert_output, _cluster_tokens, _fp8_block_gmm, _fp8_quant_per_row_block, _mask_nonlocal_experts)

def _fp8_block_scale_moe_cascade(
    expert_logits: Any,
    routing_bias: Any,
    hidden_states: Any,
    hidden_states_scale: Any,
    gemm1_weights: Any,
    gemm1_scales: Any,
    gemm2_weights: Any,
    gemm2_scales: Any,
    output: Any,
    num_experts: int,
    top_k: int,
    n_groups: int,
    top_k_groups: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling: float,
    tile_tokens_dim: int,
    routing_method_type: int,
) -> None:
    """fp8_block_scale_moe CANN 路径（950）：纯 fp8 小算子级联。

    激活 scale 布局 (H//128, S)（上游转置存放），权重 scale
    (E, N//128, K//128)；GEMM 按 K 方向 128 分块调用 fp8 grouped GEMM
    并以 block scale 折算累加；swiglu 输出按逐行 128-block 动态量化回
    fp8 供 fc2。全程 fp8 计算，无 bf16 中转、无重量化。
    """
    import torch

    from flashinfer.runtime.reference_kernels import _route_topk

    del intermediate_size, tile_tokens_dim
    idx, w = _route_topk(
        expert_logits.float(),
        routing_bias.float(),
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
    )
    w = _mask_nonlocal_experts(idx, w, local_expert_offset, local_num_experts)
    seq, hidden = hidden_states.shape
    inter = gemm1_weights.shape[1] // 2

    x_sorted, inverse, group_list, e_of_row, order = _cluster_tokens(
        hidden_states, idx, num_experts
    )
    # 激活 block scale 与聚簇后的行对齐（行方向不分块）
    xs = hidden_states_scale.float().t()  # (S, H//128)
    xs_sorted = xs.repeat_interleave(top_k, dim=0)[order].contiguous()
    y1 = _fp8_block_gmm(
        x_sorted, xs_sorted, gemm1_weights, gemm1_scales.float(),
        group_list, e_of_row,
    )
    h = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    h8, hs = _fp8_quant_per_row_block(h)
    y2 = _fp8_block_gmm(
        h8.contiguous(), hs, gemm2_weights, gemm2_scales.float(),
        group_list, e_of_row,
    )
    out = _aggregate_expert_output(y2, inverse, w, seq, top_k, hidden)
    output.copy_(out.to(output.dtype))
    return None

run = _fp8_block_scale_moe_cascade
