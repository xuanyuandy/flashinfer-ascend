"""reference implementation of gemm.mm_bf16."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.mm_bf16.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/gemm/mm_bf16_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
