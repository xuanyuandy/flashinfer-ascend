"""Backend-independent declaration for sampling.sampling_from_logits."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='sampling_from_logits', domain='sampling', op='sampling.sampling_from_logits', public_entry='flashinfer.sampling:sampling_from_logits')
