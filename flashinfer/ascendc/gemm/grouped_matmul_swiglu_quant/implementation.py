"""Ascend C descriptor identity and runner binding for gemm.grouped_matmul_swiglu_quant."""

from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.ascendc._runtime import bind_kernel
from .runner import admission, make_runner
from dataclasses import replace
from flashinfer.ascendc._catalog import load_contract

KERNEL_ID = 'ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B'
OP = 'gemm.grouped_matmul_swiglu_quant'

def descriptor():
    bound = bind_kernel(KERNEL_ID, make_runner)
    recipe = load_contract(KERNEL_ID)[1]
    return replace(bound, admission=lambda signature: admission(recipe, signature))

IMPLEMENTATION = ImplementationDefinition(KERNEL_ID, OP, "ascendc", descriptor, origin="flashinfer.ascendc")


# Author-owned native ABI and build inputs; generated JSON is derived from these fields.
from flashinfer.ascendc.definition import BuildConfig, NativeDefinition
from flashinfer.runtime import CapabilitySpec
from flashinfer.jit.recipe import TensorParameter, ShapeConstraint, BuildDependency
from flashinfer.jit.aclnn_contract import AttributeParameter

from flashinfer.runtime import CapabilityCase, CapabilitySpec, FeatureConstraint, ShapeRange
from flashinfer.operators.gemm.grouped_matmul_swiglu_quant.operator import OP, FUSION_OPS, AUXILIARY_OUTPUTS, FEATURES, quantization

def capability():
    """Authoring source for the frozen, coherent quantization/fusion cases."""
    cases = []
    for weight_dtype, scale_dtype, layout, grouped in (
        *(("int8", dtype, "NZ", False) for dtype in ("float32", "float16", "bfloat16")),
        *(("int32", "int64", layout, grouped) for layout in ("ND", "NZ") for grouped in (False, True)),
    ):
        signatures = tuple(("int8", weight_dtype, scale_dtype, "float32", assistant, "int64", "int8", "float32")
                           for assistant in (("none",) if weight_dtype == "int8" else ("float32",)))
        cases.append(CapabilityCase(
            name="%s_%s_%s_%s" % (weight_dtype, scale_dtype, layout, "group" if grouped else "channel"),
            dtype_signatures=signatures, compute_dtypes=("int8",), accumulator_dtypes=("int32",),
            shape_ranges=(ShapeRange("K", 0, 1, 16, 19984 if weight_dtype == "int32" else 65520),),
            shape_multiples=((0, 1, 16), (1, 2, 8 if weight_dtype == "int32" and layout == "NZ"
                                         else 4 if weight_dtype == "int32" else 32)),
            quantization=quantization(weight_dtype, scale_dtype, layout, 1 if grouped else None),
            quantization_group_size_ranges=((1, 1, 19999),) if grouped else (),
            fusion_ops=FUSION_OPS, auxiliary_outputs=AUXILIARY_OUTPUTS,
            features=tuple(FeatureConstraint(name, (value,)) for name, value in FEATURES)
                     + (FeatureConstraint("activation.clamp", (False, True)),),
            specializations=(("tensor_contract", ("supported",)),),
        ))
    return CapabilitySpec(OP, soc_models=("Ascend910B",), cases=tuple(cases))

NATIVE = NativeDefinition(
    name='grouped_matmul_swiglu_quant', kernel_id=KERNEL_ID, target_soc='Ascend910B',
    source_dir='csrc/kernels/ascendc/gemm/grouped_matmul_swiglu_quant',
    capability=capability(),
    build=BuildConfig(**{'device': 'op_kernel/grouped_matmul_swiglu_quant.cpp',
     'host': ('op_host/fi_grouped_matmul_swiglu_quant.cpp',),
     'op_type': 'FiGroupedMatmulSwigluQuant',
     'tiling': 'TilingGMMSwigluQuant',
     'includes': ('op_kernel', 'op_host'),
     'implementation': 'native',
     'task_type': 'aic_aiv',
     'sync_policy': 'cross_core_flags',
     'workspace_bytes': 'dynamic',
     'version': 2,
     'definitions': (),
     'compile_options': (),
     'link_options': (),
     'device_entry': 'fi_grouped_matmul_swiglu_quant',
     'host_entry': 'aclnnFiGroupedMatmulSwigluQuant',
     'launch_abi': 'aclnn.grouped_matmul_swiglu_quant.v2',
     'package_vendor': 'flashinfer_grouped_matmul_swiglu_quant',
     'recipe_id': 'grouped_matmul_swiglu_quant',
     'tiling_mode': 'host_dynamic',
     'block_dim': 1,
     'cmake': 'CMakeLists.txt'}),
    tensors=tuple(TensorParameter.from_dict(v) for v in [{'name': 'x', 'role': 'input', 'dtype': 'int8', 'shape': ['M', 'K']},
     {'name': 'weight',
      'role': 'input',
      'dtype': 'int8',
      'shape': ['E', 'K', 'P'],
      'physical_layouts': ['ND', 'NZ'],
      'native_conversions': [{'storage_dtype': 'int32',
                              'native_dtype': 'int4',
                              'element_multiplier': 8}]},
     {'name': 'weight_scale',
      'role': 'input',
      'dtype': 'float32',
      'shape': ['E', 'N'],
      'shape_alternatives': [['E', 'G', 'N']],
      'native_conversions': [{'storage_dtype': 'int64',
                              'native_dtype': 'uint64',
                              'element_multiplier': 1}]},
     {'name': 'x_scale', 'role': 'input', 'dtype': 'float32', 'shape': ['M']},
     {'name': 'weight_assistance_matrix',
      'role': 'input',
      'dtype': 'float32',
      'shape': ['E', 'N'],
      'optional': True,
      'optional_for_dtypes': [[1, 'int8']]},
     {'name': 'group_list', 'role': 'input', 'dtype': 'int64', 'shape': ['L']},
     {'name': 'y', 'role': 'output', 'dtype': 'int8', 'shape': ['M', 'D']},
     {'name': 'y_scale', 'role': 'output', 'dtype': 'float32', 'shape': ['M']}]),
    shapes=tuple(ShapeConstraint(**v) for v in [{'name': 'M', 'minimum': 1, 'maximum': 1048576, 'multiple_of': 1},
     {'name': 'K', 'minimum': 16, 'maximum': 65520, 'multiple_of': 16},
     {'name': 'E', 'minimum': 1, 'maximum': 4096, 'multiple_of': 1},
     {'name': 'P', 'minimum': 1, 'maximum': 10240, 'multiple_of': 1},
     {'name': 'N', 'minimum': 32, 'maximum': 10240, 'multiple_of': 32},
     {'name': 'D', 'minimum': 16, 'maximum': 5120, 'multiple_of': 16},
     {'name': 'G', 'minimum': 1, 'maximum': 4096, 'multiple_of': 1},
     {'name': 'L', 'minimum': 1, 'maximum': 4096, 'multiple_of': 1}]),
    dtype_signatures=(('int8', 'int8', 'float32', 'float32', 'float32', 'int64', 'int8', 'float32'), ('int8', 'int8', 'float16', 'float32', 'float32', 'int64', 'int8', 'float32'), ('int8', 'int8', 'bfloat16', 'float32', 'float32', 'int64', 'int8', 'float32'), ('int8', 'int32', 'int64', 'float32', 'float32', 'int64', 'int8', 'float32')),
    attributes=tuple(AttributeParameter(**v) for v in [{'name': 'is_enable_weight_assistance_matrix',
      'dtype': 'bool',
      'default': False,
      'minimum': None,
      'maximum': None},
     {'name': 'dequant_mode', 'dtype': 'int64', 'default': 0, 'minimum': 0, 'maximum': 1},
     {'name': 'limited',
      'dtype': 'float64',
      'default': 0.0,
      'minimum': 0.0,
      'maximum': 3.4028234663852886e+38}]),
    workspace_bound=83886080, allow_jit_fallback=False,
)

def prepare_outputs(root):
    from pathlib import Path
    from flashinfer.ascendc.authoring import prepare_native
    return prepare_native(root, Path(__file__).parent, NATIVE)
