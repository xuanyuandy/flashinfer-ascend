"""Descriptor and runner binding for gemm.segment_gemm."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/gemm/segment_gemm_v1',
     'op': 'gemm.segment_gemm',
     'backend': 'reference',
     'capability': {'op': 'gemm.segment_gemm',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/gemm/segment_gemm_v1', 'gemm.segment_gemm', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
