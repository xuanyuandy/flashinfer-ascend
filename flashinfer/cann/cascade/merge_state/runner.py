"""Execution adapter for cascade.merge_state (cann)."""

from __future__ import annotations
from flashinfer.cann.cascade._shared import (Any, Tuple)

def _merge_state_attention_update(
    v_a: Any, s_a: Any, v_b: Any, s_b: Any
) -> Tuple[Any, Any]:
    import torch_npu

    seq_len, num_heads, head_dim = v_a.shape
    out, lse_out = torch_npu.npu_attention_update(
        [s_a.float().reshape(-1), s_b.float().reshape(-1)],
        [v_a.reshape(-1, head_dim), v_b.reshape(-1, head_dim)],
        1,  # update_type=1：返回合并后的 lse
    )
    return out.reshape(seq_len, num_heads, head_dim), lse_out.reshape(
        seq_len, num_heads
    )

run = _merge_state_attention_update
