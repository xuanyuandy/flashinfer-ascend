"""Backend-independent declaration for attention.batch_decode_paged."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='batch_decode_paged', domain='attention', op='attention.batch_decode_paged', public_entry='flashinfer.decode:BatchDecodeWithPagedKVCacheWrapper')
