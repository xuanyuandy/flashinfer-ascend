"""Backend-independent declaration for attention.single_prefill."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='single_prefill', domain='attention', op='attention.single_prefill', public_entry='flashinfer.prefill:single_prefill_with_kv_cache')
