"""cascade 系列：CANN 内置算子薄封装（torch_npu / aclnn 直通，§6.4）。

对齐上游 ``flashinfer.cascade.merge_state``：底层走 CANN 已有算子
``npu_attention_update``（update_type=1 返回合并后的 lse），其计算式
``O = ΣOi·exp(lsei - lsem)`` / ``lsem = max + log Σexp(lsei - max)`` 与上游
merge_state 数学等价（数值稳定形式）。

布局适配：上游 v 为 (seq_len, num_heads, head_dim)、s 为 (seq_len, num_heads)，
算子要求 2D (batch*seq*heads, head_dim) 与 1D (batch*seq*heads)，调用前后
reshape（view，零拷贝）。
"""

from __future__ import annotations

from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

_SUPPORTED_SOCS = GENERAL_ASCEND_SOCS
