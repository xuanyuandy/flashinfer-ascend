"""Descriptor and runner binding for decode.cudnn_batch_decode_with_kv_cache."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/decode/cudnn_batch_decode_fia_v1',
     'op': 'decode.cudnn_batch_decode_with_kv_cache',
     'backend': 'cann',
     'capability': {'op': 'decode.cudnn_batch_decode_with_kv_cache',
                    'dtypes': ['float16', 'bfloat16', 'int32', 'int64'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/decode/cudnn_batch_decode_fia_v1', 'decode.cudnn_batch_decode_with_kv_cache', 'cann', descriptor, origin='flashinfer.cann.decode')
