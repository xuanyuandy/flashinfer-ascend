"""Descriptor and runner binding for norm.gemma_rmsnorm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/norm/gemma_rmsnorm_aclnn_v1',
     'op': 'norm.gemma_rmsnorm',
     'backend': 'cann',
     'capability': {'op': 'norm.gemma_rmsnorm',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/norm/gemma_rmsnorm_aclnn_v1', 'norm.gemma_rmsnorm', 'cann', descriptor, origin='flashinfer.cann.norm')
