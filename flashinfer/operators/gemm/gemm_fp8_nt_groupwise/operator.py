"""Backend-independent declaration for gemm.gemm_fp8_nt_groupwise."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='gemm_fp8_nt_groupwise', domain='gemm', op='gemm.gemm_fp8_nt_groupwise', public_entry='flashinfer.gemm:gemm_fp8_nt_groupwise')
