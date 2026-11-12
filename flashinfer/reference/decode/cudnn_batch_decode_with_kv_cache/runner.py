"""Execution adapter for decode.cudnn_batch_decode_with_kv_cache (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_cudnn_batch_decode(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    scale: float,
    max_sequence_kv: int,
    actual_seq_lens_kv: Any,
    block_tables: Any,
    out: Any,
) -> Any:
    """paged KV decode attention 高精度参考实现（float64 累计）。

    对标上游 ``flashinfer.decode.cudnn_batch_decode_with_kv_cache``：
    q (B, H_qo, D)，k/v_cache (total_pages, H_kv, page_size, D)，
    block_tables (B, num_pages_per_seq)；仅 causal（单 token 全量可见），
    支持 GQA（H_qo 为 H_kv 整数倍）。
    """
    torch = _require_torch()
    B, H_qo, D = q.shape
    _, H_kv, page_size, D_v = v_cache.shape
    group = H_qo // H_kv
    lens = actual_seq_lens_kv.long().cpu()
    tables = block_tables.long().cpu()
    with torch.no_grad():
        for b in range(B):
            kv_len = min(int(lens[b]), int(max_sequence_kv))
            pages_b = tables[b]
            # (npages, H_kv, ps, D) → (npages*ps, H_kv, D)，截断到实际长度
            k_b = (
                k_cache[pages_b]
                .double()
                .permute(0, 2, 1, 3)
                .reshape(-1, H_kv, D)[:kv_len]
            )
            v_b = (
                v_cache[pages_b]
                .double()
                .permute(0, 2, 1, 3)
                .reshape(-1, H_kv, D_v)[:kv_len]
            )
            k_rep = k_b.repeat_interleave(group, dim=1)  # (L, H_qo, D)
            v_rep = v_b.repeat_interleave(group, dim=1)
            scores = torch.einsum("hd,lhd->hl", q[b].double(), k_rep) * scale
            p = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hl,lhd->hd", p, v_rep).to(out.dtype)
    return out

run = reference_cudnn_batch_decode
