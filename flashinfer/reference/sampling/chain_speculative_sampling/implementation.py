"""reference implementation of sampling.chain_speculative_sampling."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.chain_speculative_sampling.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/sampling/chain_speculative_sampling_v1",
    support=Support(dtypes=("float16", "bfloat16", "float32"), layouts=("ND",)),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
