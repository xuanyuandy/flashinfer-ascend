"""rope 系列：CANN 内置算子薄封装（torch_npu / aclnn 直通，§6.4）。

对齐上游 ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace``：
旋转核走 CANN 已有算子 ``npu_rotary_mul``（rotary_mode=half/interleave
分别对应 GPT-NeoX / GPT-J 风格）；cos/sin 由 ``cos_sin_cache`` 按
``positions`` 索引得到（index_select，亦为 CANN 算子）。

数值验证后锁定（容器 910B + CANN 9.0 + torch_npu 2.7.1.post5）：
fp16/bf16/fp32，rotary_mode=half 与 interleave 均与 fp64 参考一致。
"""

from __future__ import annotations

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32", "int32", "int64")

_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")

def _rope_cos_sin(pos_ids: Any, rotary_dim: int, rope_scale: float, rope_theta: float):
    """按上游 pos_enc.cuh 的公式生成 cos/sin。

    ``freq[j] = rope_theta ** (-2j / rotary_dim) / rope_scale``，
    ``angle = pos * freq``（plain rope 对应 smooth=0 分支）。
    """
    import torch

    j = torch.arange(0, rotary_dim, 2, device=pos_ids.device, dtype=torch.float32)
    inv_freq = torch.pow(torch.tensor(float(rope_theta), device=pos_ids.device), -j / rotary_dim)
    inv_freq = inv_freq / float(rope_scale)
    angle = pos_ids.to(torch.float32).unsqueeze(-1) * inv_freq  # (nnz, rotary_dim/2)
    return angle.cos(), angle.sin()

def _rotate_3d(t: Any, cos_h: Any, sin_h: Any, rotary_dim: int, interleave: bool) -> Any:
    """对 ``(nnz, heads, head_dim)`` 的前 rotary_dim 维施加旋转，返回新张量。"""
    import torch_npu

    if interleave:
        # GPT-J：偶奇成对共享 cos/sin
        cos_full = cos_h.repeat_interleave(2, dim=1).unsqueeze(1)
        sin_full = sin_h.repeat_interleave(2, dim=1).unsqueeze(1)
        rotary_mode = "interleave"
    else:
        # GPT-NeoX：前后半各自旋转
        cos_full = cos_h.repeat(1, 2).unsqueeze(1)
        sin_full = sin_h.repeat(1, 2).unsqueeze(1)
        rotary_mode = "half"
    rot = t[..., :rotary_dim]
    return torch_npu.npu_rotary_mul(
        rot.contiguous(),
        cos_full.to(t.dtype),
        sin_full.to(t.dtype),
        rotary_mode=rotary_mode,
    )

def _apply_rope_pos_ids_cann(
    q: Any, k: Any, pos_ids: Any, rotary_dim, interleave: bool,
    rope_scale: float, rope_theta: float, inplace: bool,
):
    """``(nnz, heads, head_dim)`` 布局的 RoPE；in-place 与 out-of-place 共用。"""
    rd = int(rotary_dim) if rotary_dim is not None else int(q.shape[-1])
    cos_h, sin_h = _rope_cos_sin(pos_ids, rd, rope_scale, rope_theta)
    outs = []
    for t in (q, k):
        rotated = _rotate_3d(t, cos_h, sin_h, rd, interleave)
        if inplace:
            t[..., :rd].copy_(rotated)
            outs.append(t)
        else:
            o = t.clone()
            o[..., :rd].copy_(rotated)
            outs.append(o)
    return None if inplace else (outs[0], outs[1])
