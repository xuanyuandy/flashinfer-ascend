"""cann implementation of fused_moe.cutlass_fused_moe."""


from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS
from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.fused_moe.cutlass_fused_moe.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/fused_moe/cutlass_fused_moe_gmm_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "int32", "int64", "float32"),
        soc_models=GENERAL_ASCEND_SOCS,
    ),
    runner=run,
    origin="flashinfer.cann.moe",
)

descriptor = IMPLEMENTATION.bind
