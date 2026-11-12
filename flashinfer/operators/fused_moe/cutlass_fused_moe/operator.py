"""Backend-independent declaration for fused_moe.cutlass_fused_moe."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='cutlass_fused_moe', domain='fused_moe', op='fused_moe.cutlass_fused_moe', public_entry='flashinfer.fused_moe:fused_moe')
