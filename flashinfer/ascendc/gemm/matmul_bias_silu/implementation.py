"""Ascend C implementation of gemm.matmul_bias_silu."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    NativeBinding,
    BuildConfig,
)
from flashinfer.operators.author_api import Support, DimRange
from flashinfer.operators.gemm.matmul_bias_silu.operator import OPERATOR
from flashinfer.runtime import ShapeRange
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from .runner import make_runner

KERNEL_ID = "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950"
OP = "gemm.matmul_bias_silu"

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id="ascendc/t2/matmul_bias_silu/native/op_package/Ascend950",
    support=Support(
        dtypes=("bfloat16",),
        soc_models=("Ascend950",),
        specializations=(("tensor_contract", ("supported",)),),
        shape_ranges=(
            ShapeRange(name="M", tensor_index=0, axis=0, minimum=16, maximum=256),
            ShapeRange(name="K", tensor_index=0, axis=1, minimum=16, maximum=256),
            ShapeRange(name="K", tensor_index=1, axis=0, minimum=16, maximum=256),
            ShapeRange(name="N", tensor_index=1, axis=1, minimum=16, maximum=256),
            ShapeRange(name="N", tensor_index=2, axis=0, minimum=16, maximum=256),
            ShapeRange(name="M", tensor_index=3, axis=0, minimum=16, maximum=256),
            ShapeRange(name="N", tensor_index=3, axis=1, minimum=16, maximum=256),
        ),
        physical_layouts=("ND",),
    ),
    build=BuildConfig(
        device="op_kernel/native.cpp",
        host=("op_host/matmul_bias_silu.cpp",),
        op_type="FiMatmulBiasSilu",
        tiling="FiMatmulBiasSiluTiling",
        includes=("include",),
        compile_options=("-O2",),
        device_entry="fi_matmul_bias_silu",
        host_entry="aclnnFiMatmulBiasSilu",
        launch_abi="aclnn.matmul_bias_silu.v2",
        package_vendor="flashinfer_t2_matmul_bias_silu",
        recipe_id="matmul_bias_silu",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter(name="a", role="input", dtype="bfloat16", shape=("M", "K")),
            TensorParameter(name="b", role="input", dtype="bfloat16", shape=("K", "N")),
            TensorParameter(name="bias", role="input", dtype="bfloat16", shape=("N",)),
            TensorParameter(
                name="out", role="output", dtype="bfloat16", shape=("M", "N")
            ),
        ),
        dimensions=(
            ShapeConstraint(name="M", minimum=16, maximum=256, multiple_of=16),
            ShapeConstraint(name="N", minimum=16, maximum=256, multiple_of=16),
            ShapeConstraint(name="K", minimum=16, maximum=256, multiple_of=16),
        ),
    ),
    runner_factory=make_runner,
    workspace_bound=17039360,
    allow_jit_fallback=True,
)

NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
