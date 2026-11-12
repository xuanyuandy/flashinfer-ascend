"""Descriptor and runner binding for sampling.chain_speculative_sampling."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/sampling/chain_speculative_sampling_v1',
     'op': 'sampling.chain_speculative_sampling',
     'backend': 'reference',
     'capability': {'op': 'sampling.chain_speculative_sampling',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': ['ND'],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/sampling/chain_speculative_sampling_v1', 'sampling.chain_speculative_sampling', 'reference', descriptor, origin='flashinfer._sampling')
