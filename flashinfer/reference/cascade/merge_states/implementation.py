"""reference implementation of cascade.merge_states."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.cascade.merge_states.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/cascade/merge_states_v1",
    support=Support(dtypes=("float16", "bfloat16", "float32")),
    runner=run,
    origin="flashinfer._cascade",
)

descriptor = IMPLEMENTATION.bind
