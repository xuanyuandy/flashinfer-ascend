"""Descriptor and runner binding for rope.apply_rope_pos_ids_inplace."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'reference/rope/apply_rope_pos_ids_inplace_v1',
     'op': 'rope.apply_rope_pos_ids_inplace',
     'backend': 'reference',
     'capability': {'op': 'rope.apply_rope_pos_ids_inplace',
                    'dtypes': [],
                    'layouts': [],
                    'soc_models': [],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('reference/rope/apply_rope_pos_ids_inplace_v1', 'rope.apply_rope_pos_ids_inplace', 'reference', descriptor, origin='flashinfer.runtime.reference_kernels')
