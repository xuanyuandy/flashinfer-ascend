"""Descriptor and runner binding for fused_moe.trtllm_fp8_block_scale_moe."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/fused_moe/trtllm_fp8_block_scale_moe_v1',
     'op': 'fused_moe.trtllm_fp8_block_scale_moe',
     'backend': 'reference',
     'capability': {'op': 'fused_moe.trtllm_fp8_block_scale_moe',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/fused_moe/trtllm_fp8_block_scale_moe_v1', 'fused_moe.trtllm_fp8_block_scale_moe', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
