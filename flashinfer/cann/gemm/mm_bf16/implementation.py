"""Descriptor and runner binding for gemm.mm_bf16."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/gemm/mm_bf16_v1',
     'op': 'gemm.mm_bf16',
     'backend': 'cann',
     'capability': {'op': 'gemm.mm_bf16',
                    'dtypes': ['bfloat16', 'float16', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/gemm/mm_bf16_v1', 'gemm.mm_bf16', 'cann', descriptor, origin='flashinfer.cann.gemm')
