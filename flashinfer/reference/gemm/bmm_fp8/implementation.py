"""reference implementation of gemm.bmm_fp8."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.bmm_fp8.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/gemm/bmm_fp8_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
