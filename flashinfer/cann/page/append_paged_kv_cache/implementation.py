"""Descriptor and runner binding for page.append_paged_kv_cache."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/page/append_paged_kv_cache_v1',
     'op': 'page.append_paged_kv_cache',
     'backend': 'cann',
     'capability': {'op': 'page.append_paged_kv_cache',
                    'dtypes': ['float16', 'bfloat16', 'float32', 'int32', 'int64'],
                    'layouts': ['NHD', 'HND'],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': []}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/page/append_paged_kv_cache_v1', 'page.append_paged_kv_cache', 'cann', descriptor, origin='flashinfer._page')
