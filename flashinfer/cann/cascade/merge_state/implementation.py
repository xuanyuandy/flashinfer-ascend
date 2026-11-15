"""cann implementation of cascade.merge_state."""


from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS
from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.cascade.merge_state.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/cascade/merge_state_attention_update_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        soc_models=GENERAL_ASCEND_SOCS,
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
