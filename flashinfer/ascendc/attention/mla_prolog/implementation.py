"""Ascend C declaration for attention.mla_prolog.

The public tensor contract is taken from the frozen standalone host ABI
(``Dsv41MlaPrologArguments``; an implementation detail, not the public repo
name) and the frozen definition. The device entry ``fi_mla_prolog``
reconstructs the CATLASS ``UnifiedMlaPrologKernel`` Params from host tiling;
host tiling supplies workspace offsets and the hardware sync address.
A3 / arch22 / Ascend910C only.
"""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    BuildConfig,
    NativeBinding,
)
from flashinfer.operators.attention.mla_prolog.operator import (
    FUSION_OPS,
    OPERATOR,
    QUANTIZATION,
    RMS_NORM_EPS_DEFAULT,
    RMS_NORM_EPS_MAXIMUM,
    RMS_NORM_EPS_MINIMUM,
)
from flashinfer.operators.author_api import Support
from flashinfer.runtime import CapabilityCase, FeatureConstraint, ShapeRange
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import (
    BuildDependency,
    CATLASS_REVISION,
    ShapeConstraint,
    TensorParameter,
)
from .runner import make_runner

KERNEL_ID = "ascendc/attention/mla_prolog/catlass/op_package/Ascend910C"
OP = "attention.mla_prolog"

# Inputs followed by the two fresh outputs (q_output, qr), in NativeBinding order.
DTYPE_SIGNATURE = (
    "bfloat16", "bfloat16", "bfloat16", "int8", "bfloat16", "bfloat16",
    "bfloat16", "float32", "float32", "int32", "bfloat16", "bfloat16",
    "bfloat16",
)


def _tensor(name, role, dtype, shape, *, physical_layouts=("ND",)):
    return TensorParameter(
        name=name, role=role, dtype=dtype, shape=shape,
        physical_layouts=physical_layouts,
    )


IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id=KERNEL_ID,
    name="catlass",
    support=Support(
        soc_models=("Ascend910C",),
        dtypes=("bfloat16", "int8", "float32", "int32"),
        physical_layouts=("ND", "NZ"),
        shape_ranges=(
            ShapeRange(name="T", tensor_index=0, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=7, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=8, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=9, axis=0, minimum=1, maximum=4096),
            ShapeRange(name="T", tensor_index=12, axis=0, minimum=1, maximum=4096),
        ),
        cases=(CapabilityCase(
            name="a3_bf16_fixed_chain",
            dtype_signatures=(DTYPE_SIGNATURE,),
            compute_dtypes=("bfloat16",),
            accumulator_dtypes=("float32",),
            quantization=QUANTIZATION,
            fusion_ops=FUSION_OPS,
            features=(
                FeatureConstraint("normalization.kind", ("rms",)),
                FeatureConstraint("normalization.affine", (True,)),
            ),
        ),),
    ),
    build=BuildConfig(
        device="op_kernel/mla_prolog.cpp",
        host=("op_host/mla_prolog_def.cpp",),
        op_type="FiMlaProlog",
        tiling="MlaPrologTiling",
        includes=("include", "op_kernel", "op_host"),
        cmake="CMakeLists.txt",
        implementation="catlass",
        dependencies=(
            BuildDependency(name="catlass", revision=CATLASS_REVISION),
        ),
        definitions=("CATLASS_ARCH=2201",),
        compile_options=("-O2",),
        link_options=("-lascendcl",),
        workspace_bytes="dynamic",
        device_entry="fi_mla_prolog",
        host_entry="aclnnFiMlaProlog",
        launch_abi="aclnn.mla_prolog.v2",
        package_vendor="flashinfer_mla_prolog",
        recipe_id="mla_prolog",
    ),
    binding=NativeBinding(
        tensors=(
            _tensor("x", "input", "bfloat16", ("T", 5120)),
            _tensor("w_qa", "input", "bfloat16", (1280, 5120)),
            _tensor("gamma_qa", "input", "bfloat16", (1280,)),
            _tensor("w_qb", "input", "int8", (1024, 80, 16, 32),
                    physical_layouts=("NZ",)),
            _tensor("w_qb_scale", "input", "bfloat16", (32768,)),
            _tensor("w_kv", "input", "bfloat16", (512, 5120)),
            _tensor("gamma_kv", "input", "bfloat16", (512,)),
            _tensor("rope_cos", "input", "float32", ("T", 1, 1, 64)),
            _tensor("rope_sin", "input", "float32", ("T", 1, 1, 64)),
            _tensor("slot_indices", "input", "int32", ("T", 2)),
            # The kernel writes the addressed rows of this input in place.
            _tensor("kv_cache", "input", "bfloat16", (22594, 128, 512)),
            _tensor("q_output", "output", "bfloat16", ("T", 64, 512)),
            # Required second output: BF16 q_norm(wq_a(x)); the kernel writes
            # the same normBf16 row it already materializes for quantization.
            _tensor("qr", "output", "bfloat16", ("T", 1280)),
        ),
        dimensions=(ShapeConstraint(name="T", minimum=1, maximum=4096),),
        dtype_signatures=(DTYPE_SIGNATURE,),
        attributes=(
            AttributeParameter(
                "rms_norm_eps", "float64", RMS_NORM_EPS_DEFAULT,
                RMS_NORM_EPS_MINIMUM, RMS_NORM_EPS_MAXIMUM),
        ),
    ),
    runner_factory=make_runner,
    # Conservative bound for T=4096 and 24 AIC, derived from the kernel's
    # MakeWorkspaceLayout (about 99 MiB); the exact size is host-computed.
    workspace_bound=128 * 1024 * 1024,
    allow_jit_fallback=False,
)

IMPLEMENTATIONS = (IMPLEMENTATION,)
NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
