"""Backend-independent declaration for rope.apply_rope."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='apply_rope', domain='rope', op='rope.apply_rope', public_entry='flashinfer.rope:apply_rope')
