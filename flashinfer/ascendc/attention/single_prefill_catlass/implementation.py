"""Ascend C implementation of ascendc.single_prefill_catlass."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    NativeBinding,
    BuildConfig,
)
from flashinfer.operators.author_api import Support, DimRange
from flashinfer.operators.attention.single_prefill_catlass.operator import OPERATOR
from flashinfer.runtime import KernelConstraints
from flashinfer.runtime.targets import ARCH22_SOCS
from flashinfer.jit.recipe import BuildDependency, TensorParameter, CATLASS_REVISION
from .runner import make_runner

KERNEL_ID = "ascendc/t4/flash_attention/catlass/op_package/Ascend950"
OP = "ascendc.single_prefill_catlass"

def _implementation(target):
    return AscendCImplementation(
        operator=OPERATOR,
        kernel_id=KERNEL_ID.rsplit("/", 1)[0] + "/" + target,
        support=Support(
            dimensions={
                "S": DimRange(minimum=128, maximum=512, multiple_of=128),
                "T": DimRange(minimum=128, maximum=512, multiple_of=128),
                "H": DimRange(minimum=1, maximum=8, multiple_of=1),
                "D": DimRange(minimum=128, maximum=128, multiple_of=128),
            },
            dtypes=("float16", "bfloat16"),
            soc_models=(target,),
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
                    name="catlass", revision=CATLASS_REVISION
                ),
            ),
            definitions=("CATLASS_ARCH=" + ("2201" if target in ARCH22_SOCS else "3510"),),
            compile_options=("-O2",),
            link_options=("-lascendcl",) if target in ARCH22_SOCS else (),
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
        workspace_bound=256 * 1024 * 1024 if target in ARCH22_SOCS else 16 * 1024 * 1024,
        allow_jit_fallback=True,
        constraints=KernelConstraints(
            dtype_signatures=(
                ("float16", "float16", "float16", "float16"),
                ("bfloat16", "bfloat16", "bfloat16", "bfloat16"),
            )
        ),
    )


IMPLEMENTATION = _implementation("Ascend950")
_ARCH22_IMPLEMENTATIONS = tuple(_implementation(target) for target in ARCH22_SOCS)
for _variant in _ARCH22_IMPLEMENTATIONS:
    object.__setattr__(_variant, "_recipe_target_suffix", _variant.native.target_soc)
IMPLEMENTATIONS = (IMPLEMENTATION,) + _ARCH22_IMPLEMENTATIONS

NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
