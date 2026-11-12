"""Descriptor and runner binding for attention.single_prefill."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/attention/single_prefill_fia_v1',
     'op': 'attention.single_prefill',
     'backend': 'cann',
     'capability': {'op': 'attention.single_prefill',
                    'dtypes': ['float16', 'bfloat16'],
                    'layouts': ['NHD', 'HND'],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': [['head_dim', ['64', '128']]]}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/attention/single_prefill_fia_v1', 'attention.single_prefill', 'cann', descriptor, origin='flashinfer.cann.attention')
