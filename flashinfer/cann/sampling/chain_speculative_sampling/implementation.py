"""cann implementation of sampling.chain_speculative_sampling."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.sampling.chain_speculative_sampling.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/sampling/chain_speculative_sampling_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        layouts=("ND",),
        soc_models=("Ascend950",),
    ),
    runner=run,
    origin="flashinfer._sampling",
)

descriptor = IMPLEMENTATION.bind
