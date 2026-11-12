"""Execution adapter for rope.apply_rope_with_cos_sin_cache_inplace (cann)."""

from __future__ import annotations
from flashinfer.cann.rope._shared import (Any)

def _apply_rope_cos_sin_cache_aclnn(
    positions: Any,
    query: Any,
    key: Any,
    head_size: int,
    cos_sin_cache: Any,
    is_neox: bool,
) -> None:
    import torch_npu

    rotary_dim = cos_sin_cache.shape[-1]
    half = rotary_dim // 2
    cs = cos_sin_cache.index_select(0, positions)  # (nnz, rd)，float32
    cos_half, sin_half = cs[:, :half], cs[:, half:]
    if is_neox:
        # half 模式：r1/r2 与旋转维同宽，复制为 [cos|cos] / [sin|sin]
        cos_full = cos_half.repeat(1, 2).unsqueeze(1)  # (nnz, 1, rd)
        sin_full = sin_half.repeat(1, 2).unsqueeze(1)
        rotary_mode = "half"
    else:
        # interleave 模式：偶奇成对共享 cos/sin，交错展开
        cos_full = cos_half.repeat_interleave(2, dim=1).unsqueeze(1)
        sin_full = sin_half.repeat_interleave(2, dim=1).unsqueeze(1)
        rotary_mode = "interleave"
    for t in (query, key):
        t3 = t.view(t.shape[0], -1, head_size)
        rot = t3[..., :rotary_dim]
        out = torch_npu.npu_rotary_mul(
            rot.contiguous(),  # 部分旋转时切片非连续，先物化
            cos_full.to(t.dtype),
            sin_full.to(t.dtype),
            rotary_mode=rotary_mode,
        )
        rot.copy_(out)

run = _apply_rope_cos_sin_cache_aclnn
