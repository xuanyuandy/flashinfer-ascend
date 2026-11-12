"""reference implementation of gemm.gemm_fp8_nt_groupwise."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.gemm_fp8_nt_groupwise.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/gemm/gemm_fp8_nt_groupwise_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
