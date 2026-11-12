"""Descriptor and runner binding for attention.decode."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/attention/decode_v1',
     'op': 'attention.decode',
     'backend': 'reference',
     'capability': {'op': 'attention.decode',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/attention/decode_v1', 'attention.decode', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
