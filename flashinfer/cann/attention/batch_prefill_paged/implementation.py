"""cann implementation of attention.batch_prefill_paged."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.attention.batch_prefill_paged.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/attention/batch_prefill_paged_fia_v1",
    support=Support(
        dtypes=("float16", "bfloat16"),
        layouts=("NHD", "HND"),
        soc_models=("Ascend950",),
        specializations=(("head_dim", ("64", "128")),),
    ),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
