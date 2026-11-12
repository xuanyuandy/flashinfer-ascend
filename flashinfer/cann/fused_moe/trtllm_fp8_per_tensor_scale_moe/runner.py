"""Execution adapter for fused_moe.trtllm_fp8_per_tensor_scale_moe (cann)."""

from __future__ import annotations
from flashinfer.cann.fused_moe._shared import (Any, _aggregate_expert_output, _cluster_tokens, _fp8_gmm, _fp8_quant_per_row, _mask_nonlocal_experts)

def _fp8_per_tensor_scale_moe_cascade(
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
    """fp8_per_tensor_scale_moe CANN 路径（950）：纯 fp8 小算子级联。

    数据流：路由（fp32 小算子）→ token 聚簇 → fc1 fp8 grouped GEMM →
    逐专家 scale（s1×s1_gate）折算 → swiglu → 逐行动态量化回 fp8 →
    fc2 fp8 grouped GEMM → scale（s2×中间量化系数）折算 → 反置换 +
    路由权重聚合。GEMM 全程 fp8 计算，激活/权重无 bf16 中转、无重量化。
    use_routing_scales_on_input 在线性场景数学等价，按输出侧处理。
    """
    import torch

    from flashinfer.runtime.reference_kernels import _route_topk

    del intermediate_size, use_routing_scales_on_input, tile_tokens_dim
    del store_workspace_info
    idx, w = _route_topk(
        routing_logits.float(),
        routing_bias.float(),
        top_k,
        n_group,
        topk_group,
        routed_scaling_factor,
        routing_method_type,
    )
    w = _mask_nonlocal_experts(idx, w, local_expert_offset, local_num_experts)
    seq, hidden = hidden_states.shape
    inter = gemm1_weights.shape[1] // 2

    x_sorted, inverse, group_list, e_of_row, _ = _cluster_tokens(
        hidden_states, idx, num_experts
    )
    # fc1：fp8 GEMM，per-tensor scale 按行所属专家折算到输出
    w1_t = [gemm1_weights[e].t().contiguous() for e in range(num_experts)]
    y1 = _fp8_gmm(x_sorted, w1_t, group_list).float()
    c1 = output1_scales_scalar.float() * output1_scales_gate_scalar.float()
    y1 *= c1[e_of_row].unsqueeze(-1)
    # swiglu → 逐行动态量化回 fp8 供 fc2
    h = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    h8, hs = _fp8_quant_per_row(h)
    # fc2：fp8 GEMM，scale = s2 × 中间量化系数
    w2_t = [gemm2_weights[e].t().contiguous() for e in range(num_experts)]
    y2 = _fp8_gmm(h8.contiguous(), w2_t, group_list).float()
    y2 *= (output2_scales_scalar.float()[e_of_row] * hs.squeeze(-1)).unsqueeze(-1)
    out = _aggregate_expert_output(y2, inverse, w, seq, top_k, hidden)
    output.copy_(out.to(output.dtype))
    return None

run = _fp8_per_tensor_scale_moe_cascade
