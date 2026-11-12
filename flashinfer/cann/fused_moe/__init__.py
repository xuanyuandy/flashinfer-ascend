"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.fused_moe._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _FP8_E4M3_MAX, _FP8_SUPPORTED_DTYPES, _FP8_SUPPORTED_SOCS, _SUPPORTED_DTYPES, _SUPPORTED_SOCS, _aggregate_expert_output, _cluster_tokens, _fp8_block_gmm, _fp8_gmm, _fp8_quant_per_row, _fp8_quant_per_row_block, _fused_moe_megamoe, _gmm_expert_compute, _grouped_matmul, _mask_nonlocal_experts)
from flashinfer.cann.fused_moe.cutlass_fused_moe.runner import _cutlass_fused_moe_gmm
from flashinfer.cann.fused_moe.trtllm_fp8_block_scale_moe.runner import _fp8_block_scale_moe_cascade
from flashinfer.cann.fused_moe.trtllm_fp8_per_tensor_scale_moe.runner import _fp8_per_tensor_scale_moe_cascade


def moe_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.moe')
