"""Backend-independent declaration for decode.cudnn_batch_decode_with_kv_cache."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='cudnn_batch_decode_with_kv_cache', domain='decode', op='decode.cudnn_batch_decode_with_kv_cache', public_entry='flashinfer.decode:cudnn_batch_decode_with_kv_cache')
