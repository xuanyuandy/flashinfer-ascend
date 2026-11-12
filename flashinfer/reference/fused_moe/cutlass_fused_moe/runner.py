"""Execution adapter for fused_moe.cutlass_fused_moe (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _moe_expert_compute, _require_torch)

def reference_cutlass_fused_moe(
    input: Any,
    token_selected_experts: Any,
    token_final_scales: Any,
    fc1_expert_weights: Any,
    fc2_expert_weights: Any,
    output_dtype: Any,
    output: Any,
    ep_size: int = 1,
    ep_rank: int = 0,
    moe_context: Any = None,
) -> Any:
    """cutlass_fused_moe 非量化参考实现（fp32 累计，单卡语义）。"""
    if ep_size != 1:
        raise NotImplementedError(
            "reference kernel 仅单卡语义；ep_size > 1 请走 CANN megaMoe 路径"
        )
    del ep_rank, moe_context
    torch = _require_torch()
    with torch.no_grad():
        out = _moe_expert_compute(
            input.float(),
            token_selected_experts.long(),
            token_final_scales.float(),
            fc1_expert_weights.float(),
            fc2_expert_weights.float(),
            0,
            fc1_expert_weights.shape[0],
        )
        del output_dtype  # 输出精度以 output 张量 dtype 为准
        output.copy_(out.to(output.dtype))
    return output

run = reference_cutlass_fused_moe
