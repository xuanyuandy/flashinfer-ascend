"""Backend-independent declaration for sampling.chain_speculative_sampling."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='chain_speculative_sampling', domain='sampling', op='sampling.chain_speculative_sampling', public_entry='flashinfer.sampling:chain_speculative_sampling')
