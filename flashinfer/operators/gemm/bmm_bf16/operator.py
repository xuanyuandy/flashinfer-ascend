"""Backend-independent declaration for gemm.bmm_bf16."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='bmm_bf16', domain='gemm', op='gemm.bmm_bf16', public_entry='flashinfer.gemm:bmm_bf16')
