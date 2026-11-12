"""cann implementation of decode.cudnn_batch_decode_with_kv_cache."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.decode.cudnn_batch_decode_with_kv_cache.operator import (
    OPERATOR,
)
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/decode/cudnn_batch_decode_fia_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "int32", "int64"),
        soc_models=("Ascend910B", "Ascend950"),
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
