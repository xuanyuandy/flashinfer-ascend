"""Ascend C implementation of gemm.matmul."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    NativeBinding,
    BuildConfig,
)
from flashinfer.operators.author_api import Support, DimRange
from flashinfer.operators.gemm.matmul.operator import OPERATOR
from flashinfer.jit.recipe import TensorParameter
from .runner import make_runner

KERNEL_ID = "ascendc/gemm/matmul/native/op_package/Ascend950"
OP = "gemm.matmul"

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id="ascendc/gemm/matmul/native/op_package/Ascend950",
    support=Support(
        dimensions={
            "M": DimRange(minimum=16, maximum=256, multiple_of=16),
            "N": DimRange(minimum=16, maximum=256, multiple_of=16),
            "K": DimRange(minimum=16, maximum=256, multiple_of=16),
        },
        dtypes=("bfloat16",),
        soc_models=("Ascend950",),
        specializations=(("tensor_contract", ("supported",)),),
        physical_layouts=("ND",),
    ),
    build=BuildConfig(
        device="op_kernel/native.cpp",
        host=("op_host/matmul.cpp",),
        op_type="FiMatmul",
        tiling="FiMatmulTiling",
        includes=("include",),
        task_type="aic_only",
        sync_policy="compiler_default",
        workspace_bytes=0,
        compile_options=("-O2",),
        device_entry="fi_matmul",
        host_entry="aclnnFiMatmul",
        launch_abi="aclnn.matmul.v2",
        package_vendor="flashinfer_matmul",
        recipe_id="matmul",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter(name="a", role="input", dtype="bfloat16", shape=("M", "K")),
            TensorParameter(name="b", role="input", dtype="bfloat16", shape=("K", "N")),
            TensorParameter(
                name="out", role="output", dtype="bfloat16", shape=("M", "N")
            ),
        )
    ),
    runner_factory=make_runner,
)

NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
