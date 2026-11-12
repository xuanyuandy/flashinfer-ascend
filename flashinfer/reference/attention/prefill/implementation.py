"""Descriptor and runner binding for attention.prefill."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/attention/prefill_ragged_v1',
     'op': 'attention.prefill',
     'backend': 'reference',
     'capability': {'op': 'attention.prefill',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/attention/prefill_ragged_v1', 'attention.prefill', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
