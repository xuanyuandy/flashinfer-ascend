"""reference implementation of cascade.merge_state."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.cascade.merge_state.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/cascade/merge_state_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
