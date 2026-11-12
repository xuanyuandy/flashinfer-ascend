"""triton implementation of gemm.matmul."""

from flashinfer.operators.author_api import (
    TritonImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.matmul.operator import OPERATOR
from .runner import TritonMatmulRunner

IMPLEMENTATION = TritonImplementation(
    operator=OPERATOR,
    kernel_id="triton/gemm/matmul_v1/Ascend950",
    support=Support(
        dimensions={
            "M": DimRange(minimum=16, maximum=256, multiple_of=16),
            "K": DimRange(minimum=16, maximum=256, multiple_of=16),
            "N": DimRange(minimum=16, maximum=256, multiple_of=16),
        },
        dtypes=("bfloat16",),
        soc_models=("Ascend950",),
        specializations=(("tensor_contract", ("supported",)),),
        physical_layouts=("ND",),
        contiguous=True,
    ),
    runner_factory=TritonMatmulRunner,
    source=PythonKernel(),
    target_soc="Ascend950",
)

descriptor = IMPLEMENTATION.bind
