"""Backend-independent declaration for norm.gemma_fused_add_rmsnorm."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='gemma_fused_add_rmsnorm', domain='norm', op='norm.gemma_fused_add_rmsnorm', public_entry='flashinfer.norm:gemma_fused_add_rmsnorm')
