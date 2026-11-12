"""cann implementation of norm.layernorm."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.norm.layernorm.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/norm/layernorm_aclnn_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        soc_models=("Ascend910B", "Ascend950"),
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
