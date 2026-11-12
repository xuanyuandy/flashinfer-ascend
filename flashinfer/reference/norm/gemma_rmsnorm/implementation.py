"""Descriptor and runner binding for norm.gemma_rmsnorm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/norm/gemma_rmsnorm_v1',
     'op': 'norm.gemma_rmsnorm',
     'backend': 'reference',
     'capability': {'op': 'norm.gemma_rmsnorm',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/norm/gemma_rmsnorm_v1', 'norm.gemma_rmsnorm', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
