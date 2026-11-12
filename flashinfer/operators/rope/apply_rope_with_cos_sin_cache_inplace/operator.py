"""Backend-independent declaration for rope.apply_rope_with_cos_sin_cache_inplace."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='apply_rope_with_cos_sin_cache_inplace', domain='rope', op='rope.apply_rope_with_cos_sin_cache_inplace', public_entry='flashinfer.rope:apply_rope_with_cos_sin_cache_inplace')
