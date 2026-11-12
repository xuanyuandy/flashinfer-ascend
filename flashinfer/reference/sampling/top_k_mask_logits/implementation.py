"""reference implementation of sampling.top_k_mask_logits."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.top_k_mask_logits.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/sampling/top_k_mask_logits_v1",
    support=Support(dtypes=("float16", "bfloat16", "float32"), layouts=("ND",)),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
