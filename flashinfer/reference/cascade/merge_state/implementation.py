"""Descriptor and runner binding for cascade.merge_state."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/cascade/merge_state_v1',
     'op': 'cascade.merge_state',
     'backend': 'reference',
     'capability': {'op': 'cascade.merge_state',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/cascade/merge_state_v1', 'cascade.merge_state', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
