"""Ascend950 fused residual mix, coefficient update and RMSNorm projection."""

from flashinfer.operators.author_api import CatlassDslImplementation, DimRange, PythonKernel, Support
from flashinfer.operators.mhc.mega_mhc.operator import OPERATOR
from flashinfer.runtime import CapabilityCase, KernelConstraints

from .runner import CatlassDslMhcRunner


IMPLEMENTATION = CatlassDslImplementation(
    operator=OPERATOR,
    kernel_id="catlass_dsl/mhc/mega_mhc/Ascend950",
    target_soc="Ascend950",
    support=Support(
        dimensions={
            "T": DimRange(1, 4096),
            "H": DimRange(1024, 7168, 64),
            "R": DimRange(4, 4),
            "ONE": DimRange(1, 1),
            "MF": DimRange(24, 24),
            "THREE": DimRange(3, 3),
            "C": DimRange(4096, 28672, 64),
        },
        dtypes=("bfloat16", "float32"),
        soc_models=("Ascend950",),
        physical_layouts=("ND",),
        specializations=(("tensor_contract", ("supported",)),),
        cases=(CapabilityCase(
            name="fused_residual_mix",
            compute_dtypes=("float32",),
            accumulator_dtypes=("float32",),
        ),),
        contiguous=True,
    ),
    constraints=KernelConstraints(
        dtype_signatures=((
            "bfloat16", "bfloat16", "float32", "float32", "float32", "float32",
            "float32", "float32", "bfloat16", "bfloat16", "float32", "float32",
            "bfloat16", "float32",
        ),),
    ),
    source=PythonKernel(language="catlass_dsl"),
    runner_factory=CatlassDslMhcRunner,
)
