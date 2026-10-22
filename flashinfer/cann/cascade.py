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

from typing import Any, Tuple

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# v 支持 fp16/bf16/fp32（算子声明集）；s fp32 一并列入
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")


def _merge_state_attention_update(
    v_a: Any, s_a: Any, v_b: Any, s_b: Any
) -> Tuple[Any, Any]:
    import torch_npu

    seq_len, num_heads, head_dim = v_a.shape
    out, lse_out = torch_npu.npu_attention_update(
        [s_a.float().reshape(-1), s_b.float().reshape(-1)],
        [v_a.reshape(-1, head_dim), v_b.reshape(-1, head_dim)],
        1,  # update_type=1：返回合并后的 lse
    )
    return out.reshape(seq_len, num_heads, head_dim), lse_out.reshape(
        seq_len, num_heads
    )


def cascade_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN cascade 注册表。"""
    return (
        KernelDescriptor(
            kernel_id="cann/cascade/merge_state_attention_update_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="cascade.merge_state",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_merge_state_attention_update,
        ),
    )
