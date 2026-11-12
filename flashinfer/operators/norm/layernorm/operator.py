"""Backend-independent declaration for norm.layernorm."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='layernorm', domain='norm', op='norm.layernorm', public_entry='flashinfer.norm:layernorm')
