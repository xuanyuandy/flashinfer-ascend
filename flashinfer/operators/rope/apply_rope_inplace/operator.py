"""Backend-independent declaration for rope.apply_rope_inplace."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='apply_rope_inplace', domain='rope', op='rope.apply_rope_inplace', public_entry='flashinfer.rope:apply_rope_inplace')
