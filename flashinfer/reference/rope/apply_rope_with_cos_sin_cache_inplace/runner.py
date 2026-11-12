"""Execution adapter for rope.apply_rope_with_cos_sin_cache_inplace (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, _require_torch)

def reference_apply_rope_with_cos_sin_cache_inplace(
    positions: Any,
    query: Any,
    key: Any,
    head_size: int,
    cos_sin_cache: Any,
    is_neox: bool,
) -> None:
    """RoPE（cos/sin cache 变体）高精度参考实现（float64 累计，原地修改）。

    对标上游 ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace``：
    ``cos_sin_cache`` 前一半为 cos、后一半为 sin，按 ``positions`` 索引；
    仅旋转每个 head 的前 ``rotary_dim`` 维。``is_neox=True`` 为半旋转
    （GPT-NeoX 风格），``False`` 为交错旋转（GPT-J 风格）。
    """
    torch = _require_torch()
    rotary_dim = cos_sin_cache.shape[-1]
    half = rotary_dim // 2
    with torch.no_grad():
        cs = cos_sin_cache.double().index_select(0, positions)  # (nnz, rd)
        cos = cs[:, :half].unsqueeze(1)  # (nnz, 1, rd/2)
        sin = cs[:, half:].unsqueeze(1)
        for t in (query, key):
            t3 = t.view(t.shape[0], -1, head_size)
            x = t3[..., :rotary_dim].double()
            if is_neox:
                x1, x2 = x[..., :half], x[..., half:]
                rot = torch.cat(
                    [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1
                )
            else:
                xe, xo = x[..., 0::2], x[..., 1::2]
                rot = torch.stack(
                    [xe * cos - xo * sin, xo * cos + xe * sin], dim=-1
                ).flatten(-2)
            t3[..., :rotary_dim].copy_(rot.to(t.dtype))

run = reference_apply_rope_with_cos_sin_cache_inplace
