"""triton implementation of ascendc.single_prefill_catlass."""

from flashinfer.operators.author_api import (
    TritonImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.attention.single_prefill_catlass.operator import OPERATOR
from flashinfer.runtime import KernelConstraints, KernelLaunchABI
from .runner import TritonFlashAttentionRunner


def prepare_outputs(root):
    from flashinfer._resources import data_root

    return dict(
        descriptor=descriptor(),
        manifest=(data_root(root) / "kernels/triton_registry.json").relative_to(root),
        outputs={},
    )


def admission(signature):
    if signature.dtypes not in (("float16",) * 4, ("bfloat16",) * 4):
        return ("dense prefill requires homogeneous float16 or bfloat16 tensors",)
    if (
        len(signature.shapes) != 4
        or len(signature.shapes[0]) != 3
        or len(signature.shapes[1]) != 3
        or signature.shapes[0][0] % 128
        or signature.shapes[1][0] % 128
    ):
        return ("dense prefill S/T require multiple_of=128",)
    if "0" in dict(signature.specializations).get("tensor_contiguity", ""):
        return ("dense prefill requires contiguous Tensors",)
    return ()


IMPLEMENTATION = TritonImplementation(
    operator=OPERATOR,
    kernel_id="triton/attention/dense_prefill_v1/Ascend950",
    support=Support(
        dimensions={
            "S": DimRange(minimum=128, maximum=512, multiple_of=128),
            "H": DimRange(minimum=1, maximum=8, multiple_of=1),
            "D": DimRange(minimum=128, maximum=128, multiple_of=1),
            "T": DimRange(minimum=128, maximum=512, multiple_of=128),
        },
        dtypes=("float16", "bfloat16"),
        soc_models=("Ascend950",),
        specializations=(("tensor_contract", ("supported",)),),
        physical_layouts=("ND",),
    ),
    runner_factory=TritonFlashAttentionRunner,
    source=PythonKernel(),
    scope="experimental",
    constraints=KernelConstraints(
        dtype_signatures=(
            ("float16", "float16", "float16", "float16"),
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16"),
        )
    ),
    launch_abi=KernelLaunchABI(
        abi_name="triton.dense_prefill.v1",
        entry_point="_dense_prefill_kernel",
        argument_names=("q", "k", "v", "out", "S", "T", "H", "stream"),
        mutable_arguments=("out",),
    ),
    target_soc="Ascend950",
    admission=admission,
)

descriptor = IMPLEMENTATION.bind
