"""Execution adapter for fused_moe.cutlass_fused_moe (cann)."""

from __future__ import annotations
from flashinfer.cann.fused_moe._shared import (Any, _fused_moe_megamoe, _gmm_expert_compute)

def _cutlass_fused_moe_gmm(
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
    if ep_size > 1:
        del output_dtype  # 输出精度以 output 张量 dtype 为准
        return _fused_moe_megamoe(
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            output,
            moe_context,
        )
    del ep_rank

    out = _gmm_expert_compute(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
    )
    del output_dtype  # 输出精度以 output 张量 dtype 为准
    output.copy_(out.to(output.dtype))
    return output

run = _cutlass_fused_moe_gmm
