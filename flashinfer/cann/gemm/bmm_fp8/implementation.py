"""Descriptor and runner binding for gemm.bmm_fp8."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/gemm/bmm_fp8_grouped_v1',
     'op': 'gemm.bmm_fp8',
     'backend': 'cann',
     'capability': {'op': 'gemm.bmm_fp8',
                    'dtypes': ['float8_e4m3fn', 'float8_e5m2', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/gemm/bmm_fp8_grouped_v1', 'gemm.bmm_fp8', 'cann', descriptor, origin='flashinfer.cann.gemm')
