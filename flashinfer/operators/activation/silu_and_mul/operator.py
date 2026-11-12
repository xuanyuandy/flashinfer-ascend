"""Backend-independent declaration for activation.silu_and_mul."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='silu_and_mul', domain='activation', op='activation.silu_and_mul', public_entry='flashinfer.activation:silu_and_mul')
