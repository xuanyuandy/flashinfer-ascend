"""Descriptor and runner binding for gemm.bmm_fp8."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/gemm/bmm_fp8_v1',
     'op': 'gemm.bmm_fp8',
     'backend': 'reference',
     'capability': {'op': 'gemm.bmm_fp8',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/gemm/bmm_fp8_v1', 'gemm.bmm_fp8', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
