"""reference implementation of decode.cudnn_batch_decode_with_kv_cache."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.decode.cudnn_batch_decode_with_kv_cache.operator import (
    OPERATOR,
)
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/decode/cudnn_batch_decode_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
