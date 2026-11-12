"""reference implementation of sampling.top_k_sampling_from_probs."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.top_k_sampling_from_probs.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/sampling/top_k_sampling_from_probs_v1",
    support=Support(dtypes=("float16", "bfloat16", "float32"), layouts=("ND",)),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
