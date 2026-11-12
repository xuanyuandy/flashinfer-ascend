"""Backend-independent declaration for rope.apply_rope_with_cos_sin_cache."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='apply_rope_with_cos_sin_cache', domain='rope', op='rope.apply_rope_with_cos_sin_cache', public_entry='flashinfer.rope:apply_rope_with_cos_sin_cache')
