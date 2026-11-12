"""decode 系列：CANN 内置算子薄封装（torch_npu / aclnn 直通，§6.4）。

对齐上游 ``flashinfer.decode.cudnn_batch_decode_with_kv_cache``：
底层走 CANN 已有算子 ``npu_fused_infer_attention_score``（q 的 S=1 时进入
IncreFlashAttention 分支），分页 KV 经 block_table 寻址。

布局适配：上游 k/v_cache 为 (total_pages, H_kv, page_size, D)，FIA paged
分支要求 (total_pages, page_size, H_kv * D)，调用前做一次 permute+物化
（v1 正确性优先，布局转换开销随后续里程碑消除）。
"""

from __future__ import annotations

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

_SUPPORTED_DTYPES = ("float16", "bfloat16", "int32", "int64")

_SUPPORTED_SOCS = ("Ascend910B", "Ascend950")
