"""reference implementation of fused_moe.trtllm_fp8_block_scale_moe."""

from flashinfer.operators.author_api import (
    ReferenceImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.fused_moe.trtllm_fp8_block_scale_moe.operator import OPERATOR
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/fused_moe/trtllm_fp8_block_scale_moe_v1",
    support=Support(),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
