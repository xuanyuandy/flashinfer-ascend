"""Execution adapter for attention.prefill (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_repeat_kv, _require_torch)

def _prefill_ragged_ref(q, k, v, causal: bool = False, **_unused):
    """朴素 ragged prefill：q (total_q, H, D)，k/v (total_kv, H_kv, D)。"""
    torch = _require_torch()
    k, v = _repeat_kv(k, v, q.shape[-2])
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float())
    scores = scores / (q.shape[-1] ** 0.5)
    if causal:
        q_len, k_len = q.shape[0], k.shape[0]
        mask = torch.full((q_len, k_len), float("-inf"), device=scores.device)
        mask = torch.triu(mask, diagonal=k_len - q_len + 1)
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float()).to(q.dtype)

run = _prefill_ragged_ref
