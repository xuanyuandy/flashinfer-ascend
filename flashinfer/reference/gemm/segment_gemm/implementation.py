"""reference implementation of gemm.segment_gemm."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.segment_gemm.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/gemm/segment_gemm_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
