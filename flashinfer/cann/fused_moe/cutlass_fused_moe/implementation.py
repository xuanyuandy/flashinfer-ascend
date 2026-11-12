"""Descriptor and runner binding for fused_moe.cutlass_fused_moe."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/fused_moe/cutlass_fused_moe_gmm_v1',
     'op': 'fused_moe.cutlass_fused_moe',
     'backend': 'cann',
     'capability': {'op': 'fused_moe.cutlass_fused_moe',
                    'dtypes': ['float16', 'bfloat16', 'int32', 'int64', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend910_93', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/fused_moe/cutlass_fused_moe_gmm_v1', 'fused_moe.cutlass_fused_moe', 'cann', descriptor, origin='flashinfer.cann.moe')
