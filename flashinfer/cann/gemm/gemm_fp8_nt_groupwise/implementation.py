"""Descriptor and runner binding for gemm.gemm_fp8_nt_groupwise."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/gemm/gemm_fp8_nt_groupwise_block_v1',
     'op': 'gemm.gemm_fp8_nt_groupwise',
     'backend': 'cann',
     'capability': {'op': 'gemm.gemm_fp8_nt_groupwise',
                    'dtypes': ['float8_e4m3fn', 'float8_e5m2', 'float32'],
                    'layouts': [],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/gemm/gemm_fp8_nt_groupwise_block_v1', 'gemm.gemm_fp8_nt_groupwise', 'cann', descriptor, origin='flashinfer.cann.gemm')
