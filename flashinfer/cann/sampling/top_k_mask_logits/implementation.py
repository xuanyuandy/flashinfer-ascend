"""cann implementation of sampling.top_k_mask_logits."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.top_k_mask_logits.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/sampling/top_k_mask_logits_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        layouts=("ND",),
        soc_models=("Ascend950",),
    ),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
