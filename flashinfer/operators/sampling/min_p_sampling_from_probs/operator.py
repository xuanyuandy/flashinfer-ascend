"""Backend-independent declaration for sampling.min_p_sampling_from_probs."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='min_p_sampling_from_probs', domain='sampling', op='sampling.min_p_sampling_from_probs', public_entry='flashinfer.sampling:min_p_sampling_from_probs')
