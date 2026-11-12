"""Descriptor and runner binding for fused_moe.trtllm_fp8_block_scale_moe."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/fused_moe/fp8_block_scale_moe_cascade_v1',
     'op': 'fused_moe.trtllm_fp8_block_scale_moe',
     'backend': 'cann',
     'capability': {'op': 'fused_moe.trtllm_fp8_block_scale_moe',
                    'dtypes': ['float16', 'bfloat16', 'int32', 'int64', 'float32', 'float8_e4m3fn'],
                    'layouts': [],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/fused_moe/fp8_block_scale_moe_cascade_v1', 'fused_moe.trtllm_fp8_block_scale_moe', 'cann', descriptor, origin='flashinfer.cann.moe')
