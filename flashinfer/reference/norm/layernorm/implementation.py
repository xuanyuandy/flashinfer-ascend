"""reference implementation of norm.layernorm."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.norm.layernorm.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/norm/layernorm_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
