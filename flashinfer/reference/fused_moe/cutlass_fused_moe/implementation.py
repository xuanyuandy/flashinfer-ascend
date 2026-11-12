"""reference implementation of fused_moe.cutlass_fused_moe."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.fused_moe.cutlass_fused_moe.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/fused_moe/cutlass_fused_moe_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
