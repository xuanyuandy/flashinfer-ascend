"""Opt-in Triton dense prefill implementation; excluded from the production registry."""
from dataclasses import replace
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.runtime.manifest import descriptor_from_dict
from .runner import TritonFlashAttentionRunner


def descriptor():
    value = descriptor_from_dict({'kernel_id': 'triton/attention/dense_prefill_v1/Ascend950',
     'op': 'ascendc.single_prefill_catlass',
     'backend': 'triton',
     'constraints': {'supported_socs': [],
                     'dtype_signatures': [['float16', 'float16', 'float16', 'float16'],
                                          ['bfloat16', 'bfloat16', 'bfloat16', 'bfloat16']],
                     'layout_signatures': [],
                     'required_features': [],
                     'quant_storage_dtypes': [],
                     'deterministic': None},
     'workspace': {'constant_bytes': 0, 'dynamic_coefficients': [], 'alignment': 32},
     'int_workspace': {'constant_bytes': 0, 'dynamic_coefficients': [], 'alignment': 32},
     'artifact': None,
     'launch_abi': {'abi_name': 'triton.dense_prefill.v1',
                    'entry_point': '_dense_prefill_kernel',
                    'argument_names': ['q', 'k', 'v', 'out', 'S', 'T', 'H', 'stream'],
                    'mutable_arguments': ['out'],
                    'stream_argument': 'stream',
                    'pointer_width_bits': 64,
                    'metadata_schema_version': 1,
                    'schema_version': 1},
     'binary_abi': None,
     'priority': 0,
     'op_schema_version': 1,
     'tiling_schema_version': 1,
     'capability_binding': None,
     'source': {'language': 'triton',
                'locator': 'triton/attention/single_prefill_catlass/kernels.py',
                'digest': '66bfda1ca48ed1e49d0fb5ff4b5dc3956a8a59171a0bf8894d65635e3ca436e2',
                'schema_version': 1},
     'target_soc': 'Ascend950',
     'compile_mode': 'jit',
     'capability': {'op': 'ascendc.single_prefill_catlass',
                    'dtypes': ['float16', 'bfloat16'],
                    'layouts': [],
                    'soc_models': ['Ascend950'],
                    'min_aicore': 0,
                    'specializations': [['tensor_contract', ['supported']]],
                    'shape_ranges': [{'name': 'S',
                                      'tensor_index': 0,
                                      'axis': 0,
                                      'minimum': 128,
                                      'maximum': 512},
                                     {'name': 'H',
                                      'tensor_index': 0,
                                      'axis': 1,
                                      'minimum': 1,
                                      'maximum': 8},
                                     {'name': 'D',
                                      'tensor_index': 0,
                                      'axis': 2,
                                      'minimum': 128,
                                      'maximum': 128},
                                     {'name': 'T',
                                      'tensor_index': 1,
                                      'axis': 0,
                                      'minimum': 128,
                                      'maximum': 512},
                                     {'name': 'H',
                                      'tensor_index': 1,
                                      'axis': 1,
                                      'minimum': 1,
                                      'maximum': 8},
                                     {'name': 'D',
                                      'tensor_index': 1,
                                      'axis': 2,
                                      'minimum': 128,
                                      'maximum': 128},
                                     {'name': 'T',
                                      'tensor_index': 2,
                                      'axis': 0,
                                      'minimum': 128,
                                      'maximum': 512},
                                     {'name': 'H',
                                      'tensor_index': 2,
                                      'axis': 1,
                                      'minimum': 1,
                                      'maximum': 8},
                                     {'name': 'D',
                                      'tensor_index': 2,
                                      'axis': 2,
                                      'minimum': 128,
                                      'maximum': 128},
                                     {'name': 'S',
                                      'tensor_index': 3,
                                      'axis': 0,
                                      'minimum': 128,
                                      'maximum': 512},
                                     {'name': 'H',
                                      'tensor_index': 3,
                                      'axis': 1,
                                      'minimum': 1,
                                      'maximum': 8},
                                     {'name': 'D',
                                      'tensor_index': 3,
                                      'axis': 2,
                                      'minimum': 128,
                                      'maximum': 128}],
                    'physical_layouts': ['ND']}})
    return replace(value, runner=TritonFlashAttentionRunner(value), admission=admission)


IMPLEMENTATION = ImplementationDefinition(
    'triton/attention/dense_prefill_v1/Ascend950', 'ascendc.single_prefill_catlass', "triton", descriptor,
    scope="experimental", origin="flashinfer.triton.attention")


def prepare_outputs(root):
    return dict(descriptor=descriptor(), manifest="flashinfer/data/kernels/triton_registry.json", outputs={})


def admission(signature):
    if signature.dtypes not in (("float16",) * 4, ("bfloat16",) * 4):
        return ("dense prefill requires homogeneous float16 or bfloat16 tensors",)
    if (len(signature.shapes) != 4 or len(signature.shapes[0]) != 3 or len(signature.shapes[1]) != 3
            or signature.shapes[0][0] % 128 or signature.shapes[1][0] % 128):
        return ("dense prefill S/T require multiple_of=128",)
    if '0' in dict(signature.specializations).get('tensor_contiguity', ''):
        return ("dense prefill requires contiguous Tensors",)
    return ()
