"""Backend-independent declaration for attention.batch_prefill_paged."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='batch_prefill_paged', domain='attention', op='attention.batch_prefill_paged', public_entry='flashinfer.prefill:BatchPrefillWithPagedKVCacheWrapper')
