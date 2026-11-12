"""Execution adapter for decode.cudnn_batch_decode_with_kv_cache (cann)."""

from __future__ import annotations
from flashinfer.cann.decode._shared import (Any)

def _cudnn_batch_decode_fia(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    scale: float,
    max_sequence_kv: int,
    actual_seq_lens_kv: Any,
    block_tables: Any,
    out: Any,
) -> Any:
    import torch_npu

    total_pages, H_kv, page_size, _ = k_cache.shape
    B, H_qo, _ = q.shape
    # (pages, H_kv, ps, D) → (pages, ps, H_kv*D)
    k = (
        k_cache.permute(0, 2, 1, 3)
        .reshape(total_pages, page_size, H_kv * k_cache.shape[-1])
        .contiguous()
    )
    v = (
        v_cache.permute(0, 2, 1, 3)
        .reshape(total_pages, page_size, H_kv * v_cache.shape[-1])
        .contiguous()
    )
    result = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(1),  # (B, 1, H_qo, D)，BSND
        k,
        v,
        actual_seq_lengths_kv=actual_seq_lens_kv.cpu().tolist(),
        block_table=block_tables.int(),
        num_heads=H_qo,
        num_key_value_heads=H_kv,
        scale=float(scale),
        input_layout="BSND",
        block_size=page_size,
    )
    y = result[0] if isinstance(result, (tuple, list)) else result
    out.copy_(y.reshape(B, H_qo, -1))
    return out

run = _cudnn_batch_decode_fia
