"""cann implementation of gemm.gemm_fp8_nt_groupwise."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.gemm_fp8_nt_groupwise.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/gemm/gemm_fp8_nt_groupwise_block_v1",
    support=Support(
        dtypes=("float8_e4m3fn", "float8_e5m2", "float32"), soc_models=("Ascend950",)
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
