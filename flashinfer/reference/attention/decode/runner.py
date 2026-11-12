"""Execution adapter for attention.decode (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (_repeat_kv, _require_torch)

def _decode_ref(q, k_cache, v_cache, **_unused):
    """朴素单步 decode：q (B, H, D)，k_cache/v_cache (B, S, H_kv, D)。"""
    torch = _require_torch()
    k, v = _repeat_kv(k_cache, v_cache, q.shape[-2])
    scores = torch.einsum("bhd,bshd->bhs", q.float(), k.float())
    scores = scores / (q.shape[-1] ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bshd->bhd", probs, v.float()).to(q.dtype)

run = _decode_ref
