"""Descriptor and runner binding for gemm.segment_gemm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/gemm/segment_gemm_v1',
     'op': 'gemm.segment_gemm',
     'backend': 'cann',
     'capability': {'op': 'gemm.segment_gemm',
                    'dtypes': ['bfloat16', 'float16', 'float32', 'int32', 'int64'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/gemm/segment_gemm_v1', 'gemm.segment_gemm', 'cann', descriptor, origin='flashinfer.cann.gemm')
