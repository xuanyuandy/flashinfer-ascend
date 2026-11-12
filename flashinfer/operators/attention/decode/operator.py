"""Backend-independent declaration for attention.decode."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='decode', domain='attention', op='attention.decode', public_entry=None)
