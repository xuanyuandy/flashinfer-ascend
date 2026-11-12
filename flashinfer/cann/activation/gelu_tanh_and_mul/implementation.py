"""cann implementation of activation.gelu_tanh_and_mul."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.activation.gelu_tanh_and_mul.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/activation/gelu_tanh_and_mul_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        layouts=("ND",),
        soc_models=("Ascend950",),
    ),
    runner=run,
    origin="flashinfer._activation",
)

descriptor = IMPLEMENTATION.bind
