"""Ascend C descriptor identity and runner binding for gemm.matmul_bias_silu."""

from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.ascendc._runtime import bind_kernel
from .runner import make_runner

KERNEL_ID = 'ascendc/t2/matmul_bias_silu/native/op_package/Ascend950'
OP = 'gemm.matmul_bias_silu'

def descriptor():
    return bind_kernel(KERNEL_ID, make_runner)

IMPLEMENTATION = ImplementationDefinition(KERNEL_ID, OP, "ascendc", descriptor, origin="flashinfer.ascendc")


# Author-owned native ABI and build inputs; generated JSON is derived from these fields.
from flashinfer.ascendc.definition import BuildConfig, NativeDefinition
from flashinfer.runtime import CapabilitySpec
from flashinfer.jit.recipe import TensorParameter, ShapeConstraint, BuildDependency
from flashinfer.jit.aclnn_contract import AttributeParameter

NATIVE = NativeDefinition(
    name='matmul_bias_silu', kernel_id=KERNEL_ID, target_soc='Ascend950',
    source_dir='csrc/kernels/ascendc/gemm/matmul_bias_silu',
    capability=CapabilitySpec.from_dict({'op': 'gemm.matmul_bias_silu',
     'dtypes': ['bfloat16'],
     'layouts': [],
     'soc_models': ['Ascend950'],
     'min_aicore': 0,
     'specializations': [['tensor_contract', ['supported']]],
     'shape_ranges': [{'name': 'M', 'tensor_index': 0, 'axis': 0, 'minimum': 16, 'maximum': 256},
                      {'name': 'K', 'tensor_index': 0, 'axis': 1, 'minimum': 16, 'maximum': 256},
                      {'name': 'K', 'tensor_index': 1, 'axis': 0, 'minimum': 16, 'maximum': 256},
                      {'name': 'N', 'tensor_index': 1, 'axis': 1, 'minimum': 16, 'maximum': 256},
                      {'name': 'N', 'tensor_index': 2, 'axis': 0, 'minimum': 16, 'maximum': 256},
                      {'name': 'M', 'tensor_index': 3, 'axis': 0, 'minimum': 16, 'maximum': 256},
                      {'name': 'N', 'tensor_index': 3, 'axis': 1, 'minimum': 16, 'maximum': 256}],
     'physical_layouts': ['ND']}),
    build=BuildConfig(**{'device': 'op_kernel/native.cpp',
     'host': ('op_host/matmul_bias_silu.cpp',),
     'op_type': 'FiMatmulBiasSilu',
     'tiling': 'FiMatmulBiasSiluTiling',
     'includes': ('include',),
     'implementation': 'native',
     'task_type': 'aic_aiv',
     'sync_policy': 'cross_core_flags',
     'workspace_bytes': 'dynamic',
     'version': 2,
     'definitions': (),
     'compile_options': ('-O2',),
     'link_options': (),
     'device_entry': 'fi_matmul_bias_silu',
     'host_entry': 'aclnnFiMatmulBiasSilu',
     'launch_abi': 'aclnn.matmul_bias_silu.v2',
     'package_vendor': 'flashinfer_t2_matmul_bias_silu',
     'recipe_id': 'matmul_bias_silu',
     'tiling_mode': 'host_dynamic',
     'block_dim': 1}),
    tensors=tuple(TensorParameter.from_dict(v) for v in [{'name': 'a', 'role': 'input', 'dtype': 'bfloat16', 'shape': ['M', 'K']},
     {'name': 'b', 'role': 'input', 'dtype': 'bfloat16', 'shape': ['K', 'N']},
     {'name': 'bias', 'role': 'input', 'dtype': 'bfloat16', 'shape': ['N']},
     {'name': 'out', 'role': 'output', 'dtype': 'bfloat16', 'shape': ['M', 'N']}]),
    shapes=tuple(ShapeConstraint(**v) for v in [{'name': 'M', 'minimum': 16, 'maximum': 256, 'multiple_of': 16},
     {'name': 'N', 'minimum': 16, 'maximum': 256, 'multiple_of': 16},
     {'name': 'K', 'minimum': 16, 'maximum': 256, 'multiple_of': 16}]),
    dtype_signatures=(),
    attributes=tuple(AttributeParameter(**v) for v in []),
    workspace_bound=17039360, allow_jit_fallback=True,
)

def prepare_outputs(root):
    from pathlib import Path
    from flashinfer.ascendc.authoring import prepare_native
    return prepare_native(root, Path(__file__).parent, NATIVE)
