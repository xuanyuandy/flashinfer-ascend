"""reference implementation of activation.silu_and_mul."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.activation.silu_and_mul.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/activation/silu_and_mul_v1",
    support=Support(dtypes=("float16", "bfloat16", "float32"), layouts=("ND",)),
    runner=run,
    origin="flashinfer._activation",
)

descriptor = IMPLEMENTATION.bind
