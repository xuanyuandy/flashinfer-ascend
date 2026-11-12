"""Backend-independent declaration for sampling.sampling_from_probs."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='sampling_from_probs', domain='sampling', op='sampling.sampling_from_probs', public_entry='flashinfer.sampling:sampling_from_probs')
