"""cann implementation of gemm.matmul."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.matmul.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/gemm/matmul_v1",
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
    runner=run,
)

descriptor = IMPLEMENTATION.bind
