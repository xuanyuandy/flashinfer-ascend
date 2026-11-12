"""cann implementation of cascade.merge_states."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.cascade.merge_states.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/cascade/merge_states_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"), soc_models=("Ascend950",)
    ),
    runner=run,
    origin="flashinfer._cascade",
)

descriptor = IMPLEMENTATION.bind
