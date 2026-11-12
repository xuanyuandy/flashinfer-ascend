"""reference implementation of attention.prefill."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.attention.prefill.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/attention/prefill_ragged_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
