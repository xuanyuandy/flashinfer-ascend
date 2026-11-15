"""cann implementation of rope.apply_rope_inplace."""


from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS
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
        soc_models=GENERAL_ASCEND_SOCS,
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
