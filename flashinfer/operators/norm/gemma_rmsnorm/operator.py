"""Backend-independent declaration for norm.gemma_rmsnorm."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='gemma_rmsnorm', domain='norm', op='norm.gemma_rmsnorm', public_entry='flashinfer.norm:gemma_rmsnorm')
