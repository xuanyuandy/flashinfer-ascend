"""cann implementation of gemm.bmm_bf16."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.bmm_bf16.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/gemm/bmm_bf16_v1",
    support=Support(
        dtypes=("bfloat16", "float16", "float32"),
        soc_models=("Ascend910B", "Ascend950"),
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
