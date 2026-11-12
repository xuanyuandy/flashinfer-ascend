"""reference implementation of attention.decode."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.attention.decode.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/attention/decode_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
