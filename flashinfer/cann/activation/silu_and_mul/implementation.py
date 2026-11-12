"""Descriptor and runner binding for activation.silu_and_mul."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/activation/silu_and_mul_v1',
     'op': 'activation.silu_and_mul',
     'backend': 'cann',
     'capability': {'op': 'activation.silu_and_mul',
                    'dtypes': ['float16', 'bfloat16', 'float32'],
                    'layouts': ['ND'],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/activation/silu_and_mul_v1', 'activation.silu_and_mul', 'cann', descriptor, origin='flashinfer._activation')
