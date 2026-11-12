"""Ascend C implementation of ascendc.single_prefill_catlass."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    NativeBinding,
    BuildConfig,
)
from flashinfer.operators.author_api import Support, DimRange
from flashinfer.operators.attention.single_prefill_catlass.operator import OPERATOR
from flashinfer.runtime import KernelConstraints
from flashinfer.jit.recipe import BuildDependency, TensorParameter
from .runner import make_runner

KERNEL_ID = "ascendc/t4/flash_attention/catlass/op_package/Ascend950"
OP = "ascendc.single_prefill_catlass"

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id="ascendc/t4/flash_attention/catlass/op_package/Ascend950",
    support=Support(
        dimensions={
            "S": DimRange(minimum=128, maximum=512, multiple_of=128),
            "T": DimRange(minimum=128, maximum=512, multiple_of=128),
            "H": DimRange(minimum=1, maximum=8, multiple_of=1),
            "D": DimRange(minimum=128, maximum=128, multiple_of=128),
        },
        dtypes=("float16", "bfloat16"),
        soc_models=("Ascend950",),
        specializations=(("tensor_contract", ("supported",)),),
        physical_layouts=("ND",),
    ),
    build=BuildConfig(
        device="op_kernel/catlass.cpp",
        host=("op_host/flash_attention.cpp",),
        op_type="FiFlashAttention",
        tiling="FiFlashAttentionTiling",
        includes=("op_kernel", "op_host", "include"),
        implementation="catlass",
        dependencies=(
            BuildDependency(
                name="catlass", revision="41bf90da655bba3c66d0acd7e00abe33960ecfd6"
            ),
        ),
        definitions=("CATLASS_ARCH=3510",),
        compile_options=("-O2",),
        device_entry="fi_flash_attention",
        host_entry="aclnnFiFlashAttention",
        launch_abi="aclnn.flash_attention.v2",
        package_vendor="flashinfer_t4_flash_attention",
        recipe_id="flash_attention",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter(
                name="q", role="input", dtype="float16", shape=("S", "H", "D")
            ),
            TensorParameter(
                name="k", role="input", dtype="float16", shape=("T", "H", "D")
            ),
            TensorParameter(
                name="v", role="input", dtype="float16", shape=("T", "H", "D")
            ),
            TensorParameter(
                name="out", role="output", dtype="float16", shape=("S", "H", "D")
            ),
        ),
        dtype_signatures=(
            ("float16", "float16", "float16", "float16"),
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16"),
        ),
    ),
    runner_factory=make_runner,
    source_dir="csrc/kernels/attention/flash_attention",
    workspace_bound=16777216,
    allow_jit_fallback=True,
    constraints=KernelConstraints(
        dtype_signatures=(
            ("float16", "float16", "float16", "float16"),
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16"),
        )
    ),
)

NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
