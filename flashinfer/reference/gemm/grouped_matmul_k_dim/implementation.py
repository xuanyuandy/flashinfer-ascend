"""reference implementation of gemm.grouped_matmul_k_dim."""

from flashinfer.operators.author_api import ReferenceImplementation, Support
from flashinfer.operators.gemm.grouped_matmul_k_dim.operator import OPERATOR, capability_cases
from .runner import run

IMPLEMENTATION = ReferenceImplementation(
    operator=OPERATOR,
    kernel_id="reference/gemm/grouped_matmul_k_dim_v1",
    # No SoC restriction: this is plain torch. The cases are the same coherent
    # dtype/size clauses the native backend declares, so the explicit
    # InferenceSpec stays admissible.
    support=Support(cases=capability_cases()),
    runner=run,
)

descriptor = IMPLEMENTATION.bind
