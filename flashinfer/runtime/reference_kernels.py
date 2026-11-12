"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.reference._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _moe_expert_compute, _reference_rope_core, _repeat_kv, _require_torch, _route_topk)
from flashinfer.reference.attention.decode.runner import _decode_ref
from flashinfer.reference.attention.prefill.runner import _prefill_ragged_ref
from flashinfer.reference.cascade.merge_state.runner import reference_merge_state
from flashinfer.reference.decode.cudnn_batch_decode_with_kv_cache.runner import reference_cudnn_batch_decode
from flashinfer.reference.fused_moe.cutlass_fused_moe.runner import reference_cutlass_fused_moe
from flashinfer.reference.fused_moe.trtllm_fp8_block_scale_moe.runner import reference_trtllm_fp8_block_scale_moe
from flashinfer.reference.fused_moe.trtllm_fp8_per_tensor_scale_moe.runner import reference_trtllm_fp8_per_tensor_scale_moe
from flashinfer.reference.gemm.bmm_bf16.runner import reference_bmm_bf16
from flashinfer.reference.gemm.bmm_fp8.runner import reference_bmm_fp8
from flashinfer.reference.gemm.gemm_fp8_nt_groupwise.runner import reference_gemm_fp8_nt_groupwise
from flashinfer.reference.gemm.mm_bf16.runner import reference_mm_bf16
from flashinfer.reference.gemm.segment_gemm.runner import reference_segment_gemm
from flashinfer.reference.norm.fused_add_rmsnorm.runner import reference_fused_add_rmsnorm
from flashinfer.reference.norm.gemma_fused_add_rmsnorm.runner import reference_gemma_fused_add_rmsnorm
from flashinfer.reference.norm.gemma_rmsnorm.runner import reference_gemma_rmsnorm
from flashinfer.reference.norm.layernorm.runner import reference_layernorm
from flashinfer.reference.norm.rmsnorm.runner import reference_rmsnorm
from flashinfer.reference.rope.apply_rope_inplace.runner import reference_apply_rope_pos_ids_inplace
from flashinfer.reference.rope.apply_rope_pos_ids.runner import reference_apply_rope_pos_ids
from flashinfer.reference.rope.apply_rope_with_cos_sin_cache.runner import reference_apply_rope_with_cos_sin_cache
from flashinfer.reference.rope.apply_rope_with_cos_sin_cache_inplace.runner import reference_apply_rope_with_cos_sin_cache_inplace


def builtin_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='reference', origin='flashinfer.runtime.reference_kernels')
