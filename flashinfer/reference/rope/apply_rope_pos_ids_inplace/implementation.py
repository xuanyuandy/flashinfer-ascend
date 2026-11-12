"""reference implementation of rope.apply_rope_pos_ids_inplace."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.rope.apply_rope_pos_ids_inplace.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/rope/apply_rope_pos_ids_inplace_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
