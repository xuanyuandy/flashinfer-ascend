"""Descriptor and runner binding for sampling.top_k_mask_logits."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/sampling/top_k_mask_logits_v1',
     'op': 'sampling.top_k_mask_logits',
     'backend': 'reference',
     'capability': {'op': 'sampling.top_k_mask_logits',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': ['ND'],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/sampling/top_k_mask_logits_v1', 'sampling.top_k_mask_logits', 'reference', descriptor, origin='flashinfer._sampling')
