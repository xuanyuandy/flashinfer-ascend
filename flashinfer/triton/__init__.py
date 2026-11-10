"""Triton backend：基于 triton-ascend 的自定义 kernel 集合（§5.2）。

定位：补齐 CANN 未覆盖的算子。本包的 descriptor 注册**不依赖** triton
运行时——导入安全（host 可注册/调度）；仅 kernel 真正执行时才要求环境
内安装 triton（triton-ascend）并可用 NPU。
"""

from __future__ import annotations

from typing import Tuple

from ..runtime.schema import KernelDescriptor


def triton_descriptors() -> Tuple[KernelDescriptor, ...]:
    """Triton backend 的内置 kernel 注册表。"""
    from .norm import norm_descriptors
    from .gemm import gemm_descriptors

    return norm_descriptors() + gemm_descriptors()
