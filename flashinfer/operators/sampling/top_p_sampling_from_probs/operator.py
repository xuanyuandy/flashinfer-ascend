"""Backend-independent declaration for sampling.top_p_sampling_from_probs."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='top_p_sampling_from_probs', domain='sampling', op='sampling.top_p_sampling_from_probs', public_entry='flashinfer.sampling:top_p_sampling_from_probs')
