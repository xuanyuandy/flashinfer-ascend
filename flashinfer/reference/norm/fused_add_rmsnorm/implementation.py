"""Descriptor and runner binding for norm.fused_add_rmsnorm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/norm/fused_add_rmsnorm_v1',
     'op': 'norm.fused_add_rmsnorm',
     'backend': 'reference',
     'capability': {'op': 'norm.fused_add_rmsnorm',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/norm/fused_add_rmsnorm_v1', 'norm.fused_add_rmsnorm', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
