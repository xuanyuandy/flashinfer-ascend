"""cann implementation of page.append_paged_kv_cache."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.page.append_paged_kv_cache.operator import OPERATOR
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/page/append_paged_kv_cache_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32", "int32", "int64"),
        layouts=("NHD", "HND"),
        soc_models=("Ascend950",),
    ),
    runner=run,
    origin="flashinfer._page",
)

descriptor = IMPLEMENTATION.bind
