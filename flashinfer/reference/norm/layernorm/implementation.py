"""Descriptor and runner binding for norm.layernorm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/norm/layernorm_v1',
     'op': 'norm.layernorm',
     'backend': 'reference',
     'capability': {'op': 'norm.layernorm',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/norm/layernorm_v1', 'norm.layernorm', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
