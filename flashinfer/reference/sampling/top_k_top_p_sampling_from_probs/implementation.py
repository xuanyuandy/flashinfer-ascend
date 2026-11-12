"""Descriptor and runner binding for sampling.top_k_top_p_sampling_from_probs."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/sampling/top_k_top_p_sampling_from_probs_v1',
     'op': 'sampling.top_k_top_p_sampling_from_probs',
     'backend': 'reference',
     'capability': {'op': 'sampling.top_k_top_p_sampling_from_probs',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': ['ND'],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/sampling/top_k_top_p_sampling_from_probs_v1', 'sampling.top_k_top_p_sampling_from_probs', 'reference', descriptor, origin='flashinfer._sampling')
