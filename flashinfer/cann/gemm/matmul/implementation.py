"""Descriptor and runner binding for gemm.matmul."""

from dataclasses import replace
from functools import partial
from flashinfer.operators.gemm._shared import metadata_admission
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/gemm/matmul_v1',
     'op': 'gemm.matmul',
     'backend': 'cann',
     'capability': {'op': 'gemm.matmul',
                    'dtypes': ['bfloat16'],
                    'layouts': [],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': [['tensor_contract', ['supported']]],
                    'shape_ranges': [{'name': 'M',
                                      'tensor_index': 0,
                                      'axis': 0,
                                      'minimum': 16,
                                      'maximum': 256},
                                     {'name': 'K',
                                      'tensor_index': 0,
                                      'axis': 1,
                                      'minimum': 16,
                                      'maximum': 256},
                                     {'name': 'K',
                                      'tensor_index': 1,
                                      'axis': 0,
                                      'minimum': 16,
                                      'maximum': 256},
                                     {'name': 'N',
                                      'tensor_index': 1,
                                      'axis': 1,
                                      'minimum': 16,
                                      'maximum': 256},
                                     {'name': 'M',
                                      'tensor_index': 2,
                                      'axis': 0,
                                      'minimum': 16,
                                      'maximum': 256},
                                     {'name': 'N',
                                      'tensor_index': 2,
                                      'axis': 1,
                                      'minimum': 16,
                                      'maximum': 256}],
                    'physical_layouts': ['ND']}}), runner=run, admission=partial(metadata_admission, multiple=16, contiguous=True))


IMPLEMENTATION = ImplementationDefinition('cann/gemm/matmul_v1', 'gemm.matmul', 'cann', descriptor, origin='flashinfer.cann.gemm')
