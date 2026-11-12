"""Descriptor and runner binding for cascade.merge_states."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/cascade/merge_states_v1',
     'op': 'cascade.merge_states',
     'backend': 'reference',
     'capability': {'op': 'cascade.merge_states',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/cascade/merge_states_v1', 'cascade.merge_states', 'reference', descriptor, origin='flashinfer._cascade')
