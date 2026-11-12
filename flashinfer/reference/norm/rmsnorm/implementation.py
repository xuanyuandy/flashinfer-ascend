"""Descriptor and runner binding for norm.rmsnorm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/norm/rmsnorm_v1',
     'op': 'norm.rmsnorm',
     'backend': 'reference',
     'capability': {'op': 'norm.rmsnorm',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/norm/rmsnorm_v1', 'norm.rmsnorm', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
