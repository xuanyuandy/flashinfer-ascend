"""reference implementation of norm.rmsnorm."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.norm.rmsnorm.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/norm/rmsnorm_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
