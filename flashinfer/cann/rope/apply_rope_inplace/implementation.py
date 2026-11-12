"""cann implementation of rope.apply_rope_inplace."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.rope.apply_rope_inplace.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/rope/apply_rope_inplace_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32", "int32", "int64"),
        soc_models=("Ascend910B", "Ascend950"),
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
