"""Execution adapter for fused_moe.trtllm_fp8_per_tensor_scale_moe (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _moe_expert_compute, _require_torch, _route_topk)

def reference_trtllm_fp8_per_tensor_scale_moe(
    routing_logits: Any,
    routing_bias: Any,
    hidden_states: Any,
    gemm1_weights: Any,
    output1_scales_scalar: Any,
    output1_scales_gate_scalar: Any,
    gemm2_weights: Any,
    output2_scales_scalar: Any,
    output: Any,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    use_routing_scales_on_input: bool,
    tile_tokens_dim: int,
    routing_method_type: int,
    store_workspace_info: bool,
) -> None:
    """trtllm_fp8_per_tensor_scale_moe 参考实现（fp8 反量化到 fp32 计算）。

    per-tensor scale 折算进权重：W_eff = W_fp8 * s1 * s1_gate（gemm1）/
    W_fp8 * s2（gemm2）；use_routing_scales_on_input 决定路由权重作用在
    gemm 输入还是输出，fp32 参考下两者数学等价，直接按输出侧处理。
    """
    del intermediate_size, use_routing_scales_on_input, tile_tokens_dim
    del store_workspace_info
    torch = _require_torch()
    idx, w = _route_topk(
        routing_logits.float(),
        routing_bias.float(),
        int(top_k),
        int(n_group),
        int(topk_group),
        float(routed_scaling_factor),
        int(routing_method_type),
    )
    with torch.no_grad():
        s1 = output1_scales_scalar.float() * output1_scales_gate_scalar.float()
        w1_eff = gemm1_weights.float() * s1.reshape(-1, 1, 1)
        w2_eff = gemm2_weights.float() * output2_scales_scalar.float().reshape(-1, 1, 1)
        out = _moe_expert_compute(
            hidden_states.float(),
            idx,
            w,
            w1_eff,
            w2_eff,
            int(local_expert_offset),
            int(local_num_experts),
        )
        output.copy_(out.to(output.dtype))
    return None

run = reference_trtllm_fp8_per_tensor_scale_moe
