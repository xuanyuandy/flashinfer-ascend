"""Ascend C implementation of gemm.grouped_matmul_swiglu_quant."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    NativeBinding,
    BuildConfig,
)
from flashinfer.operators.author_api import Support, DimRange
from flashinfer.operators.gemm.grouped_matmul_swiglu_quant.operator import (
    OPERATOR,
    FUSION_OPS,
    AUXILIARY_OUTPUTS,
    FEATURES,
    quantization,
)
from flashinfer.runtime import CapabilityCase, FeatureConstraint, QuantSpec, ShapeRange
from flashinfer.runtime.targets import ARCH22_SOCS
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.jit.aclnn_contract import AttributeParameter, TensorDtypeConversion
from .runner import make_runner, admission

KERNEL_ID = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B"
OP = "gemm.grouped_matmul_swiglu_quant"


def _support():
    """Authoring source for the frozen, coherent quantization/fusion cases."""
    cases = []
    for weight_dtype, scale_dtype, layout, grouped in (
        *(("int8", dtype, "NZ", False) for dtype in ("float32", "float16", "bfloat16")),
        *(
            ("int32", "int64", layout, grouped)
            for layout in ("ND", "NZ")
            for grouped in (False, True)
        ),
    ):
        signatures = tuple(
            (
                "int8",
                weight_dtype,
                scale_dtype,
                "float32",
                assistant,
                "int64",
                "int8",
                "float32",
            )
            for assistant in (("none",) if weight_dtype == "int8" else ("float32",))
        )
        cases.append(
            CapabilityCase(
                name="%s_%s_%s_%s"
                % (
                    weight_dtype,
                    scale_dtype,
                    layout,
                    "group" if grouped else "channel",
                ),
                dtype_signatures=signatures,
                compute_dtypes=("int8",),
                accumulator_dtypes=("int32",),
                shape_ranges=(
                    ShapeRange(
                        "K", 0, 1, 16, 19984 if weight_dtype == "int32" else 65520
                    ),
                ),
                shape_multiples=(
                    (0, 1, 16),
                    (
                        1,
                        2,
                        (
                            8
                            if weight_dtype == "int32" and layout == "NZ"
                            else 4 if weight_dtype == "int32" else 32
                        ),
                    ),
                ),
                quantization=quantization(
                    weight_dtype, scale_dtype, layout, 1 if grouped else None
                ),
                quantization_group_size_ranges=((1, 1, 19999),) if grouped else (),
                fusion_ops=FUSION_OPS,
                auxiliary_outputs=AUXILIARY_OUTPUTS,
                features=tuple(
                    FeatureConstraint(name, (value,)) for name, value in FEATURES
                )
                + (FeatureConstraint("activation.clamp", (False, True)),),
                specializations=(("tensor_contract", ("supported",)),),
            )
        )
    return Support(soc_models=ARCH22_SOCS, cases=tuple(cases))


# Native tiling bounds workspace use to 80 MiB.
MAX_WORKSPACE_BYTES = 80 * 1024 * 1024

IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    name="native/op_package",
    default_target="Ascend910B",
    support=_support(),
    build=BuildConfig(
        device="op_kernel/grouped_matmul_swiglu_quant.cpp",
        host=("op_host/fi_grouped_matmul_swiglu_quant.cpp",),
        op_type="FiGroupedMatmulSwigluQuant",
        tiling="TilingGMMSwigluQuant",
        cmake="CMakeLists.txt",
        device_entry="fi_grouped_matmul_swiglu_quant",
        host_entry="aclnnFiGroupedMatmulSwigluQuant",
        launch_abi="aclnn.grouped_matmul_swiglu_quant.v2",
        package_vendor="flashinfer_grouped_matmul_swiglu_quant",
        recipe_id="grouped_matmul_swiglu_quant",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter(name="x", role="input", dtype="int8", shape=("M", "K")),
            TensorParameter(
                name="weight",
                role="input",
                dtype="int8",
                shape=("E", "K", "P"),
                physical_layouts=("ND", "NZ"),
                native_conversions=(
                    TensorDtypeConversion(
                        storage_dtype="int32", native_dtype="int4", element_multiplier=8
                    ),
                ),
            ),
            TensorParameter(
                name="weight_scale",
                role="input",
                dtype="float32",
                shape=("E", "N"),
                shape_alternatives=(("E", "G", "N"),),
                native_conversions=(
                    TensorDtypeConversion(storage_dtype="int64", native_dtype="uint64"),
                ),
            ),
            TensorParameter(
                name="x_scale", role="input", dtype="float32", shape=("M",)
            ),
            TensorParameter(
                name="weight_assistance_matrix",
                role="input",
                dtype="float32",
                shape=("E", "N"),
                optional=True,
                optional_for_dtypes=((1, "int8"),),
            ),
            TensorParameter(
                name="group_list", role="input", dtype="int64", shape=("L",)
            ),
            TensorParameter(name="y", role="output", dtype="int8", shape=("M", "D")),
            TensorParameter(
                name="y_scale", role="output", dtype="float32", shape=("M",)
            ),
        ),
        dimensions=(
            ShapeConstraint(name="M", minimum=1, maximum=1048576),
            ShapeConstraint(name="K", minimum=16, maximum=65520, multiple_of=16),
            ShapeConstraint(name="E", minimum=1, maximum=4096),
            ShapeConstraint(name="P", minimum=1, maximum=10240),
            ShapeConstraint(name="N", minimum=32, maximum=10240, multiple_of=32),
            ShapeConstraint(name="D", minimum=16, maximum=5120, multiple_of=16),
            ShapeConstraint(name="G", minimum=1, maximum=4096),
            ShapeConstraint(name="L", minimum=1, maximum=4096),
        ),
        dtype_signatures=(
            (
                "int8",
                "int8",
                "float32",
                "float32",
                "float32",
                "int64",
                "int8",
                "float32",
            ),
            (
                "int8",
                "int8",
                "float16",
                "float32",
                "float32",
                "int64",
                "int8",
                "float32",
            ),
            (
                "int8",
                "int8",
                "bfloat16",
                "float32",
                "float32",
                "int64",
                "int8",
                "float32",
            ),
            (
                "int8",
                "int32",
                "int64",
                "float32",
                "float32",
                "int64",
                "int8",
                "float32",
            ),
        ),
        attributes=(
            AttributeParameter(
                name="is_enable_weight_assistance_matrix", dtype="bool", default=False
            ),
            AttributeParameter(
                name="dequant_mode", dtype="int64", default=0, minimum=0, maximum=1
            ),
            AttributeParameter(
                name="limited",
                dtype="float64",
                default=0.0,
                minimum=0.0,
                maximum=3.4028234663852886e38,
            ),
        ),
    ),
    runner_factory=make_runner,
    workspace_bound=MAX_WORKSPACE_BYTES,
    admission=admission,
)

IMPLEMENTATION = IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind


def capability():
    return NATIVE.capability
