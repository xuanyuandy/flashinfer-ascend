"""Backend-independent declaration for attention.batch_prefill_ragged."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='batch_prefill_ragged', domain='attention', op='attention.batch_prefill_ragged', public_entry='flashinfer.prefill:BatchPrefillWithRaggedKVCacheWrapper')
