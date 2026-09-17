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

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# q/k/v 支持 fp16/bf16（FIA 声明集）；block_tables int32、
# actual_seq_lens_kv int32/int64 一并列入（capability 检查覆盖签名全部 dtype）
_SUPPORTED_DTYPES = ("float16", "bfloat16", "int32", "int64")
_SUPPORTED_SOCS = ("Ascend910B",)


def _cudnn_batch_decode_fia(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    scale: float,
    max_sequence_kv: int,
    actual_seq_lens_kv: Any,
    block_tables: Any,
    out: Any,
) -> Any:
    import torch_npu

    total_pages, H_kv, page_size, _ = k_cache.shape
    B, H_qo, _ = q.shape
    # (pages, H_kv, ps, D) → (pages, ps, H_kv*D)
    k = (
        k_cache.permute(0, 2, 1, 3)
        .reshape(total_pages, page_size, H_kv * k_cache.shape[-1])
        .contiguous()
    )
    v = (
        v_cache.permute(0, 2, 1, 3)
        .reshape(total_pages, page_size, H_kv * v_cache.shape[-1])
        .contiguous()
    )
    result = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(1),  # (B, 1, H_qo, D)，BSND
        k,
        v,
        actual_seq_lengths_kv=actual_seq_lens_kv.cpu().tolist(),
        block_table=block_tables.int(),
        num_heads=H_qo,
        num_key_value_heads=H_kv,
        scale=float(scale),
        input_layout="BSND",
        block_size=page_size,
    )
    y = result[0] if isinstance(result, (tuple, list)) else result
    out.copy_(y.reshape(B, H_qo, -1))
    return out


def decode_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN decode 注册表。"""
    return (
        KernelDescriptor(
            kernel_id="cann/decode/cudnn_batch_decode_fia_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="decode.cudnn_batch_decode_with_kv_cache",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_cudnn_batch_decode_fia,
        ),
    )
