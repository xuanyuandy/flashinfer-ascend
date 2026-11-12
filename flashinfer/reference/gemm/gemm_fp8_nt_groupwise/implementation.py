"""Descriptor and runner binding for gemm.gemm_fp8_nt_groupwise."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/gemm/gemm_fp8_nt_groupwise_v1',
     'op': 'gemm.gemm_fp8_nt_groupwise',
     'backend': 'reference',
     'capability': {'op': 'gemm.gemm_fp8_nt_groupwise',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/gemm/gemm_fp8_nt_groupwise_v1', 'gemm.gemm_fp8_nt_groupwise', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
