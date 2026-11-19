"""Ascend C implementation of gemm.grouped_matmul_k_dim."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    NativeBinding,
    BuildConfig,
)
from flashinfer.operators.author_api import Support
from flashinfer.operators.gemm.grouped_matmul_k_dim.operator import OPERATOR, capability_cases
from flashinfer.runtime.targets import ARCH22_SOCS
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from .runner import make_runner

KERNEL_ID = "ascendc/gemm/grouped_matmul_k_dim/native/op_package/Ascend910B"
OP = "gemm.grouped_matmul_k_dim"

# The tiler reserves a fixed system workspace; nothing is sized per shape.
MAX_WORKSPACE_BYTES = 16 * 1024 * 1024
MAX_STRIDE = 65535
# The host tiler stores M/N/K and the group count as uint32.
MAX_TILING_DIM = 2**31 - 1
MAX_GROUPS = 1048576

# Same native code serves both padded targets: Ascend910B (A2) and Ascend910C (A3)
# share dav-2201 and one arch22 source tree.
_TARGET_SOCS = ARCH22_SOCS


def _support():
    """The arch22 kernel accepts only ND operands; cases come from the logical contract."""
    return Support(
        soc_models=_TARGET_SOCS,
        physical_layouts=("ND",),
        cases=capability_cases(),
    )


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    name="native/op_package",
    default_target="Ascend910B",
    support=_support(),
    build=BuildConfig(
        device="op_kernel/grouped_matmul_k_dim.cpp",
        host=("op_host/fi_grouped_matmul_k_dim.cpp",),
        op_type="FiGroupedMatmulKDim",
        tiling="TilingGroupedMatmulKDim",
        cmake="CMakeLists.txt",
        device_entry="fi_grouped_matmul_k_dim",
        host_entry="aclnnFiGroupedMatmulKDim",
        launch_abi="aclnn.grouped_matmul_k_dim.v2",
        package_vendor="flashinfer_grouped_matmul_k_dim",
        recipe_id="grouped_matmul_k_dim",
        task_type="aic_only",
        sync_policy="compiler_default",
        workspace_bytes=MAX_WORKSPACE_BYTES,
    ),
    binding=NativeBinding(
        tensors=(
            # Symbols stay unshared on the grouped axis because trans_a moves K
            # between rows and columns; infer() enforces the K agreement instead.
            TensorParameter(name="x", role="input", dtype="float16", shape=("XA", "XB")),
            TensorParameter(name="weight", role="input", dtype="float16", shape=("WA", "N")),
            TensorParameter(name="group_list", role="input", dtype="int64", shape=("L",)),
            TensorParameter(name="y", role="output", dtype="float16", shape=("G", "YM", "N")),
        ),
        dimensions=(
            # Every symbol needs exactly one bound. XA/WA/YM follow the transposed
            # state, so their bound is only the uint32 tiling ceiling; XB is the
            # x row stride in both states and N is shared by weight and output.
            ShapeConstraint(name="XA", minimum=1, maximum=MAX_TILING_DIM),
            ShapeConstraint(name="XB", minimum=1, maximum=MAX_STRIDE),
            ShapeConstraint(name="WA", minimum=1, maximum=MAX_TILING_DIM),
            ShapeConstraint(name="N", minimum=1, maximum=MAX_STRIDE),
            ShapeConstraint(name="L", minimum=1, maximum=MAX_GROUPS),
            ShapeConstraint(name="G", minimum=1, maximum=MAX_GROUPS),
            ShapeConstraint(name="YM", minimum=1, maximum=MAX_TILING_DIM),
        ),
        dtype_signatures=(
            ("float16", "float16", "int64", "float16"),
            ("float16", "float16", "int64", "float32"),
            ("bfloat16", "bfloat16", "int64", "bfloat16"),
            ("bfloat16", "bfloat16", "int64", "float32"),
        ),
        attributes=(AttributeParameter(name="trans_a", dtype="bool", default=False),),
    ),
    runner_factory=make_runner,
    workspace_bound=MAX_WORKSPACE_BYTES,
)

IMPLEMENTATION = IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind


def capability():
    return NATIVE.capability
