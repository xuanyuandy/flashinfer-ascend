"""Descriptor and runner binding for attention.batch_prefill_paged."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/attention/batch_prefill_paged_fia_v1',
     'op': 'attention.batch_prefill_paged',
     'backend': 'cann',
     'capability': {'op': 'attention.batch_prefill_paged',
                    'dtypes': ['float16', 'bfloat16'],
                    'layouts': ['NHD', 'HND'],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': [['head_dim', ['64', '128']]]}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/attention/batch_prefill_paged_fia_v1', 'attention.batch_prefill_paged', 'cann', descriptor, origin='flashinfer.cann.attention')
