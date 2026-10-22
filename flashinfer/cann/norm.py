"""CANN aclnn RMSNorm 系列 kernel 封装（设计文档 §6.4）。

runner 延迟导入 torch_npu：模块保持 torch-free，host 单测可注册、
可调度，仅实际 run 时才要求 NPU 环境（§14.1）。

aclnn 语义（910B3 + CANN 9.0 实测）：
- ``npu_rms_norm(x, w, epsilon)`` → ``(y, rstd)``，
  ``y = x / sqrt(mean(x^2) + eps) * w``；
- ``npu_add_rms_norm(x1, x2, w, epsilon)`` → ``(y, rstd, x_out)``，
  ``x_out = x1 + x2``，``y = rmsnorm(x_out) * w``——
  与 flashinfer ``fused_add_rmsnorm`` 的原地语义一致（结果 copy_ 回去）。
"""

from __future__ import annotations

from typing import Any, Tuple

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# 已验证支持集：910B 与 950；扩展前先补数值验证（§11 声明须属实）
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")


def _rmsnorm_aclnn(out: Any, input: Any, weight: Any, eps: float) -> None:
    import torch_npu

    y, _rstd = torch_npu.npu_rms_norm(input, weight, epsilon=eps)
    out.copy_(y)


def _fused_add_rmsnorm_aclnn(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    import torch_npu

    y, _rstd, new_residual = torch_npu.npu_add_rms_norm(
        input, residual, weight, epsilon=eps
    )
    input.copy_(y)
    residual.copy_(new_residual)


def _gemma_rmsnorm_aclnn(out: Any, input: Any, weight: Any, eps: float) -> None:
    import torch_npu

    y, _rstd = torch_npu.npu_gemma_rms_norm(input, weight, epsilon=eps)
    out.copy_(y)


def _layernorm_aclnn(
    out: Any, input: Any, weight: Any, bias: Any, eps: float
) -> None:
    import torch.nn.functional as F

    # npu_layer_norm_eval 仅有旧式实现；A5 要求 ACLNN 路径。
    y = F.layer_norm(
        input, [input.shape[-1]], weight=weight, bias=bias, eps=eps
    )
    out.copy_(y)


def norm_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN norm 注册表（M1 首批 4 个 kernel）。"""
    return (
        KernelDescriptor(
            kernel_id="cann/norm/rmsnorm_aclnn_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="norm.rmsnorm",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_rmsnorm_aclnn,
        ),
        KernelDescriptor(
            kernel_id="cann/norm/fused_add_rmsnorm_aclnn_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="norm.fused_add_rmsnorm",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_fused_add_rmsnorm_aclnn,
        ),
        KernelDescriptor(
            kernel_id="cann/norm/gemma_rmsnorm_aclnn_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="norm.gemma_rmsnorm",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_gemma_rmsnorm_aclnn,
        ),
        KernelDescriptor(
            kernel_id="cann/norm/layernorm_aclnn_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="norm.layernorm",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_layernorm_aclnn,
        ),
    )
