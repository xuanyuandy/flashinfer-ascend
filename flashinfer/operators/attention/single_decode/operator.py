"""Backend-independent declaration for attention.single_decode."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='single_decode', domain='attention', op='attention.single_decode', public_entry='flashinfer.decode:single_decode_with_kv_cache')
