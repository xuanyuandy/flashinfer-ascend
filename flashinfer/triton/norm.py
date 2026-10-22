"""Triton norm kernels 的 descriptor 注册与薄封装（导入安全）。

本模块 **不直接 import triton**：descriptor/runner 仅声明依赖，真正的
Triton kernel 在 ``_norm_kernels.py`` 中定义并在首次执行时延迟导入，
未安装 triton-ascend 的环境仍可完成注册与调度决策。
"""

from __future__ import annotations

from typing import Any, Tuple

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# 数值验证后锁定（fp32 累计，最后一维归一化）
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")


def _gemma_fused_add_rmsnorm_triton(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    from . import _norm_kernels

    _norm_kernels.gemma_fused_add_rmsnorm(input, residual, weight, eps)


def norm_descriptors() -> Tuple[KernelDescriptor, ...]:
    """Triton norm 注册表（补齐 CANN 缺失算子）。"""
    return (
        KernelDescriptor(
            kernel_id="triton/norm/gemma_fused_add_rmsnorm_v1",
            backend=Backend.TRITON,
            compile_mode=CompileMode.JIT,
            capability=CapabilitySpec(
                op="norm.gemma_fused_add_rmsnorm",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_gemma_fused_add_rmsnorm_triton,
        ),
    )
