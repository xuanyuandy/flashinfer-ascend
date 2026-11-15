"""triton implementation of norm.gemma_fused_add_rmsnorm."""


from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS
from flashinfer.operators.author_api import (
    TritonImplementation,
    Support,
    DimRange,
    PythonKernel,
)
from flashinfer.operators.norm.gemma_fused_add_rmsnorm.operator import OPERATOR
from .runner import _gemma_fused_add_rmsnorm_triton

IMPLEMENTATION = TritonImplementation(
    operator=OPERATOR,
    kernel_id="triton/norm/gemma_fused_add_rmsnorm_v1",
    support=Support(
        dtypes=("float16", "bfloat16", "float32"),
        soc_models=GENERAL_ASCEND_SOCS,
    ),
    runner=_gemma_fused_add_rmsnorm_triton,
)

descriptor = IMPLEMENTATION.bind
