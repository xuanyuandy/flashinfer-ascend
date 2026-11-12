"""Descriptor and runner binding for attention.single_decode."""

from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import run


def descriptor():
    return replace(descriptor_from_dict({'kernel_id': 'cann/attention/single_decode_fia_v1',
     'op': 'attention.single_decode',
     'backend': 'cann',
     'graph_capability': 'needs_param_update',
     'graph_param_updates': ['actual_seq_lengths_kv'],
     'graph_replay': {'validated_kernel_id': 'cann/attention/single_decode_fia_v1',
                      'validation_tests': ['tests/test_replay_param_npu.py::test_fia_full_capture_replay_needs_parameter_update'],
                      'validated_environment': 'Ascend950PR_9579; torch 2.10.0+cpu; torch_npu '
                                               '2.10.0.post7; CANN 9.2',
                      'strategy': 'host_kwargs',
                      'parameter_sources': [['actual_seq_lengths_kv', 'kv_seq_lengths']],
                      'required_arguments': ['workspace', 'out'],
                      'fixed_arguments': [['input_layout', 'BSND'],
                                          ['atten_mask', None],
                                          ['num_heads', 8],
                                          ['num_key_value_heads', 2],
                                          ['scale', 0.125],
                                          ['softmax_lse_flag', True]],
                      'tensor_metadata': [['query', [1, 1, 8, 64], ['float16', 'bfloat16']],
                                          ['key', [1, 1024, 2, 64], ['float16', 'bfloat16']],
                                          ['value', [1, 1024, 2, 64], ['float16', 'bfloat16']]],
                      'parameter_ranges': [['actual_seq_lengths_kv', 1, 1024]]},
     'capability': {'op': 'attention.single_decode',
                    'dtypes': ['float16', 'bfloat16'],
                    'layouts': ['NHD', 'HND'],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': [['head_dim', ['64', '128']]]}}), runner=run)


IMPLEMENTATION = ImplementationDefinition('cann/attention/single_decode_fia_v1', 'attention.single_decode', 'cann', descriptor, origin='flashinfer.cann.attention')
