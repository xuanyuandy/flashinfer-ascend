"""cann implementation of norm.fused_add_rmsnorm."""


from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS
from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.norm.fused_add_rmsnorm.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/norm/fused_add_rmsnorm_aclnn_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        soc_models=GENERAL_ASCEND_SOCS,
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
