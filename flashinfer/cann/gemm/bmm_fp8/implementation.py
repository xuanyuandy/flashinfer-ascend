"""cann implementation of gemm.bmm_fp8."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.bmm_fp8.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/gemm/bmm_fp8_grouped_v1",
    support=Support(
        dtypes=("float8_e4m3fn", "float8_e5m2", "float32"), soc_models=("Ascend950",)
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
