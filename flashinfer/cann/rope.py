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

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# 数值验证后锁定；positions 为 int32/int64、cos_sin_cache 为 float32，
# 一并列入声明（capability 检查覆盖签名全部 dtype）
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32", "int32", "int64")
_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")


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


def rope_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN rope 注册表。"""
    return (
        KernelDescriptor(
            kernel_id="cann/rope/apply_rope_cos_sin_cache_aclnn_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="rope.apply_rope_with_cos_sin_cache_inplace",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_apply_rope_cos_sin_cache_aclnn,
        ),
    )
