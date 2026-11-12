"""cann implementation of fused_moe.trtllm_fp8_per_tensor_scale_moe."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.fused_moe.trtllm_fp8_per_tensor_scale_moe.operator import (
    OPERATOR,
)
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "int32", "int64", "float32", "float8_e4m3fn"),
        soc_models=("Ascend950",),
    ),
    runner=run,
    origin="flashinfer.cann.moe",
)

descriptor = IMPLEMENTATION.bind
