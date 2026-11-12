"""cann implementation of attention.single_decode."""

from flashinfer.operators.author_api import (
    CANNImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.attention.single_decode.operator import OPERATOR
from flashinfer.runtime import GraphCapability
from flashinfer.cann.attention._shared import _DECODE_REPLAY
from .runner import run

IMPLEMENTATION = CANNImplementation(
    operator=OPERATOR,
    kernel_id="cann/attention/single_decode_fia_v1",
    support=Support(
        dtypes=("float16", "bfloat16"),
        layouts=("NHD", "HND"),
        soc_models=("Ascend950",),
        specializations=(("head_dim", ("64", "128")),),
    ),
    runner=run,
    graph_capability=GraphCapability.NEEDS_PARAM_UPDATE,
    graph_param_updates=("actual_seq_lengths_kv",),
    graph_replay=_DECODE_REPLAY,
)

descriptor = IMPLEMENTATION.bind
