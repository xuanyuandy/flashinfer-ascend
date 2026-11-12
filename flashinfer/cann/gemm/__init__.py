"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.cann.gemm._shared import (Any, Backend, CapabilitySpec, CompileMode, KernelDescriptor, Tuple, _GENERAL_SOCS, _SUPPORTED_DTYPES, _SUPPORTED_SOCS, _fp8_grouped_gemm)
from flashinfer.cann.gemm.bmm_bf16.runner import _bmm_bf16_cann
from flashinfer.cann.gemm.bmm_fp8.runner import _bmm_fp8_cann
from flashinfer.cann.gemm.gemm_fp8_nt_groupwise.runner import _gemm_fp8_nt_groupwise_cann
from flashinfer.cann.gemm.matmul.runner import _matmul_cann
from flashinfer.cann.gemm.mm_bf16.runner import _mm_bf16_cann
from flashinfer.cann.gemm.segment_gemm.runner import _segment_gemm_cann


def gemm_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend='cann', origin='flashinfer.cann.gemm')
