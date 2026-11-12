"""Descriptor and runner binding for decode.cudnn_batch_decode_with_kv_cache."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/decode/cudnn_batch_decode_v1',
     'op': 'decode.cudnn_batch_decode_with_kv_cache',
     'backend': 'reference',
     'capability': {'op': 'decode.cudnn_batch_decode_with_kv_cache',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/decode/cudnn_batch_decode_v1', 'decode.cudnn_batch_decode_with_kv_cache', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
