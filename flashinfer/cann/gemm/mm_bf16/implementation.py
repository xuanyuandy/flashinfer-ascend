"""cann implementation of gemm.mm_bf16."""


from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS
from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.gemm.mm_bf16.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/gemm/mm_bf16_v1",
    support=Support(
        dtypes=("bfloat16", "float16", "float32"),
        soc_models=GENERAL_ASCEND_SOCS,
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
