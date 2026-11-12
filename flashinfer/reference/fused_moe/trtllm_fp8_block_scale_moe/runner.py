"""Execution adapter for fused_moe.trtllm_fp8_block_scale_moe (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _moe_expert_compute, _require_torch, _route_topk)

def reference_trtllm_fp8_block_scale_moe(
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
    """trtllm_fp8_block_scale_moe 参考实现（按 128-block 反量化到 fp32）。

    hidden_states_scale 布局为 (hidden//128, seq)（上游转置存放）；
    gemm1_scales (E, 2I//128, H//128)、gemm2_scales (E, H//128, I//128)。
    """
    del intermediate_size, tile_tokens_dim
    torch = _require_torch()
    idx, w = _route_topk(
        expert_logits.float(),
        routing_bias.float(),
        int(top_k),
        int(n_groups),
        int(top_k_groups),
        float(routed_scaling),
        int(routing_method_type),
    )

    def _dequant_block(x_f, s_f):
        # x (..., R, C)，s (..., R//128, C//128)，两个维度按 128-block 展开
        return x_f * s_f.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)

    with torch.no_grad():
        seq, hidden = hidden_states.shape
        xs = hidden_states_scale.float().t()  # (seq, hidden//128)，行方向不分块
        x_eff = hidden_states.float() * xs.repeat_interleave(128, dim=-1)
        w1_eff = _dequant_block(gemm1_weights.float(), gemm1_scales.float())
        w2_eff = _dequant_block(gemm2_weights.float(), gemm2_scales.float())
        out = _moe_expert_compute(
            x_eff,
            idx,
            w,
            w1_eff,
            w2_eff,
            int(local_expert_offset),
            int(local_num_experts),
        )
        output.copy_(out.to(output.dtype))
    return None

run = reference_trtllm_fp8_block_scale_moe
