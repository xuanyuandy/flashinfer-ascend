"""Ascend C descriptor identity and runner binding for ascendc.single_prefill_catlass."""

from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.ascendc._runtime import bind_kernel
from .runner import make_runner

KERNEL_ID = 'ascendc/t4/flash_attention/catlass/op_package/Ascend950'
OP = 'ascendc.single_prefill_catlass'

def descriptor():
    return bind_kernel(KERNEL_ID, make_runner)

IMPLEMENTATION = ImplementationDefinition(KERNEL_ID, OP, "ascendc", descriptor, origin="flashinfer.ascendc")


# Author-owned native ABI and build inputs; generated JSON is derived from these fields.
from flashinfer.ascendc.definition import BuildConfig, NativeDefinition
from flashinfer.runtime import CapabilitySpec
from flashinfer.jit.recipe import TensorParameter, ShapeConstraint, BuildDependency
from flashinfer.jit.aclnn_contract import AttributeParameter

NATIVE = NativeDefinition(
    name='flash_attention', kernel_id=KERNEL_ID, target_soc='Ascend950',
    source_dir='csrc/kernels/ascendc/attention/flash_attention',
    capability=CapabilitySpec.from_dict({'op': 'ascendc.single_prefill_catlass',
     'dtypes': ['float16', 'bfloat16'],
     'layouts': [],
     'soc_models': ['Ascend950'],
     'min_aicore': 0,
     'specializations': [['tensor_contract', ['supported']]],
     'shape_ranges': [{'name': 'S', 'tensor_index': 0, 'axis': 0, 'minimum': 128, 'maximum': 512},
                      {'name': 'H', 'tensor_index': 0, 'axis': 1, 'minimum': 1, 'maximum': 8},
                      {'name': 'D', 'tensor_index': 0, 'axis': 2, 'minimum': 128, 'maximum': 128},
                      {'name': 'T', 'tensor_index': 1, 'axis': 0, 'minimum': 128, 'maximum': 512},
                      {'name': 'H', 'tensor_index': 1, 'axis': 1, 'minimum': 1, 'maximum': 8},
                      {'name': 'D', 'tensor_index': 1, 'axis': 2, 'minimum': 128, 'maximum': 128},
                      {'name': 'T', 'tensor_index': 2, 'axis': 0, 'minimum': 128, 'maximum': 512},
                      {'name': 'H', 'tensor_index': 2, 'axis': 1, 'minimum': 1, 'maximum': 8},
                      {'name': 'D', 'tensor_index': 2, 'axis': 2, 'minimum': 128, 'maximum': 128},
                      {'name': 'S', 'tensor_index': 3, 'axis': 0, 'minimum': 128, 'maximum': 512},
                      {'name': 'H', 'tensor_index': 3, 'axis': 1, 'minimum': 1, 'maximum': 8},
                      {'name': 'D', 'tensor_index': 3, 'axis': 2, 'minimum': 128, 'maximum': 128}],
     'physical_layouts': ['ND']}),
    build=BuildConfig(**{'device': 'op_kernel/catlass.cpp',
     'host': ('op_host/flash_attention.cpp',),
     'op_type': 'FiFlashAttention',
     'tiling': 'FiFlashAttentionTiling',
     'includes': ('op_kernel', 'op_host', 'include'),
     'implementation': 'catlass',
     'task_type': 'aic_aiv',
     'sync_policy': 'cross_core_flags',
     'workspace_bytes': 'dynamic',
     'version': 2,
     'definitions': ('CATLASS_ARCH=3510',),
     'compile_options': ('-O2',),
     'link_options': (),
     'device_entry': 'fi_flash_attention',
     'host_entry': 'aclnnFiFlashAttention',
     'launch_abi': 'aclnn.flash_attention.v2',
     'package_vendor': 'flashinfer_t4_flash_attention',
     'recipe_id': 'flash_attention',
     'tiling_mode': 'host_dynamic',
     'block_dim': 1}, dependencies=tuple(BuildDependency(**v) for v in [{'name': 'catlass',
  'revision': '41bf90da655bba3c66d0acd7e00abe33960ecfd6',
  'include_dir': 'include'}])),
    tensors=tuple(TensorParameter.from_dict(v) for v in [{'name': 'q', 'role': 'input', 'dtype': 'float16', 'shape': ['S', 'H', 'D']},
     {'name': 'k', 'role': 'input', 'dtype': 'float16', 'shape': ['T', 'H', 'D']},
     {'name': 'v', 'role': 'input', 'dtype': 'float16', 'shape': ['T', 'H', 'D']},
     {'name': 'out', 'role': 'output', 'dtype': 'float16', 'shape': ['S', 'H', 'D']}]),
    shapes=tuple(ShapeConstraint(**v) for v in [{'name': 'S', 'minimum': 128, 'maximum': 512, 'multiple_of': 128},
     {'name': 'T', 'minimum': 128, 'maximum': 512, 'multiple_of': 128},
     {'name': 'H', 'minimum': 1, 'maximum': 8, 'multiple_of': 1},
     {'name': 'D', 'minimum': 128, 'maximum': 128, 'multiple_of': 128}]),
    dtype_signatures=(('float16', 'float16', 'float16', 'float16'), ('bfloat16', 'bfloat16', 'bfloat16', 'bfloat16')),
    attributes=tuple(AttributeParameter(**v) for v in []),
    workspace_bound=16777216, allow_jit_fallback=True,
    descriptor_metadata={'kernel_id': 'ascendc/t4/flash_attention/catlass/op_package/Ascend950',
     'op': 'ascendc.single_prefill_catlass',
     'backend': 'ascendc',
     'constraints': {'supported_socs': [],
                     'dtype_signatures': [['float16', 'float16', 'float16', 'float16'],
                                          ['bfloat16', 'bfloat16', 'bfloat16', 'bfloat16']],
                     'layout_signatures': [],
                     'required_features': [],
                     'quant_storage_dtypes': [],
                     'deterministic': None},
     'workspace': {'constant_bytes': 16777216, 'dynamic_coefficients': [], 'alignment': 32},
     'int_workspace': {'constant_bytes': 0, 'dynamic_coefficients': [], 'alignment': 32},
     'artifact': None,
     'launch_abi': {'abi_name': 'aclnn.flash_attention.v2',
                    'entry_point': 'aclnnFiFlashAttention',
                    'argument_names': ['workspace', 'workspace_bytes', 'executor', 'stream'],
                    'mutable_arguments': [],
                    'stream_argument': 'stream',
                    'pointer_width_bits': 64,
                    'metadata_schema_version': 1,
                    'schema_version': 1},
     'binary_abi': None,
     'priority': 0,
     'op_schema_version': 1,
     'tiling_schema_version': 2,
     'capability_binding': None,
     'target_soc': 'Ascend950',
     'compile_mode': 'aot'},
)

def prepare_outputs(root):
    from pathlib import Path
    from flashinfer.ascendc.authoring import prepare_native
    return prepare_native(root, Path(__file__).parent, NATIVE)
