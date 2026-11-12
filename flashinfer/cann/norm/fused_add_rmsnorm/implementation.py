"""Descriptor and runner binding for norm.fused_add_rmsnorm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/norm/fused_add_rmsnorm_aclnn_v1',
     'op': 'norm.fused_add_rmsnorm',
     'backend': 'cann',
     'capability': {'op': 'norm.fused_add_rmsnorm',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/norm/fused_add_rmsnorm_aclnn_v1', 'norm.fused_add_rmsnorm', 'cann', descriptor, origin='flashinfer.cann.norm')
