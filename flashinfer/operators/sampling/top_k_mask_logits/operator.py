"""Backend-independent declaration for sampling.top_k_mask_logits."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='top_k_mask_logits', domain='sampling', op='sampling.top_k_mask_logits', public_entry='flashinfer.sampling:top_k_mask_logits')
