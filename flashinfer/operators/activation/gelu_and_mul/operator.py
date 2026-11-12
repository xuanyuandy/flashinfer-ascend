"""Backend-independent declaration for activation.gelu_and_mul."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='gelu_and_mul', domain='activation', op='activation.gelu_and_mul', public_entry='flashinfer.activation:gelu_and_mul')
