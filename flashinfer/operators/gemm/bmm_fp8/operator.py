"""Backend-independent declaration for gemm.bmm_fp8."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='bmm_fp8', domain='gemm', op='gemm.bmm_fp8', public_entry='flashinfer.gemm:bmm_fp8')
