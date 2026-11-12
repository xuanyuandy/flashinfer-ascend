"""Backend-independent declaration for sampling.top_k_top_p_sampling_from_probs."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='top_k_top_p_sampling_from_probs', domain='sampling', op='sampling.top_k_top_p_sampling_from_probs', public_entry='flashinfer.sampling:top_k_top_p_sampling_from_probs')
