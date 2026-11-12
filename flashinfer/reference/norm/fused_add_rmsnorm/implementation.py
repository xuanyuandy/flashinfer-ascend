"""reference implementation of norm.fused_add_rmsnorm."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.norm.fused_add_rmsnorm.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/norm/fused_add_rmsnorm_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
