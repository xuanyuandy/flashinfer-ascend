"""cann implementation of sampling.top_p_sampling_from_probs."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.top_p_sampling_from_probs.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/sampling/top_p_sampling_from_probs_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        layouts=("ND",),
        soc_models=("Ascend950",),
    ),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
