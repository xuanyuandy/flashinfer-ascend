"""CANN backend 适配（设计文档 §6.4）。

职责：把 CANN 内置算子 / aclnn 组合包装成 ``KernelDescriptor`` 注册进
registry（backend=cann，compile_mode=none），与自研 kernel 走同一套
过滤 + 排序流程（§5.3）。

M1 起接入 aclnn 算子：当前覆盖 norm 系列（rmsnorm / fused_add_rmsnorm /
gemma_rmsnorm / layernorm）、rope（apply_rope_with_cos_sin_cache）、
decode（cudnn_batch_decode_with_kv_cache，走 FIA paged 分支）、
fused_moe（cutlass_fused_moe 非量化路径，走 npu_grouped_matmul；
trtllm fp8 两接口为 950 纯 fp8 小算子级联）、cascade（merge_state，
走 npu_attention_update）与 gemm（bmm_fp8 / gemm_fp8_nt_groupwise，
950 fp8 grouped GEMM 路径）。
"""

from __future__ import annotations

from typing import Tuple

from ..runtime.schema import KernelDescriptor
from .cascade import cascade_descriptors
from .decode import decode_descriptors
from .gemm import gemm_descriptors
from .moe import moe_descriptors
from .norm import norm_descriptors
from .rope import rope_descriptors


def cann_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN backend 的内置 descriptor 注册表。"""
    return (
        norm_descriptors()
        + rope_descriptors()
        + decode_descriptors()
        + moe_descriptors()
        + cascade_descriptors()
        + gemm_descriptors()
    )
