"""Descriptor and runner binding for rope.apply_rope_pos_ids."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/rope/apply_rope_pos_ids_v1',
     'op': 'rope.apply_rope_pos_ids',
     'backend': 'cann',
     'capability': {'op': 'rope.apply_rope_pos_ids',
                    'dtypes': ['float16', 'bfloat16', 'float32', 'int32', 'int64'],
                    'layouts': [],
                    'soc_models': ['Ascend910B', 'Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/rope/apply_rope_pos_ids_v1', 'rope.apply_rope_pos_ids', 'cann', descriptor, origin='flashinfer.cann.rope')
