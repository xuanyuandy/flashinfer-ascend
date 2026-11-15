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

from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

_SUPPORTED_SOCS = GENERAL_ASCEND_SOCS
