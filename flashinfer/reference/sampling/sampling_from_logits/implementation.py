"""reference implementation of sampling.sampling_from_logits."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.sampling_from_logits.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/sampling/sampling_from_logits_v1",
    support=Support(dtypes=("float16", "bfloat16", "float32"), layouts=("ND",)),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
