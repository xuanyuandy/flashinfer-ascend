"""Backend-independent declaration for attention.prefill."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='prefill', domain='attention', op='attention.prefill', public_entry=None)
