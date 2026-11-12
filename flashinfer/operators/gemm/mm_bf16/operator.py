"""Backend-independent declaration for gemm.mm_bf16."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='mm_bf16', domain='gemm', op='gemm.mm_bf16', public_entry='flashinfer.gemm:mm_bf16')
