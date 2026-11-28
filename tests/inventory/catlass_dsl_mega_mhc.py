"""Reviewed hardware and shape limits for the fused MHC CATLASS DSL kernel."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec, InputSignature
from tests.inventory import ReviewedKernel, shape_ranges


KERNELS = (
    ReviewedKernel(
        "catlass_dsl/mhc/mega_mhc/Ascend950",
        CapabilitySpec(
            op="mhc.mega_mhc",
            dtypes=("bfloat16", "float32"),
            soc_models=("Ascend950",),
            physical_layouts=("ND",),
            specializations=(("tensor_contract", ("supported",)),),
            cases=(CapabilityCase(
                name="fused_residual_mix",
                compute_dtypes=("float32",),
                accumulator_dtypes=("float32",),
            ),),
            shape_ranges=shape_ranges(
                (
                    ("T", "H"), ("T", "R", "H"), ("T", "R"), ("T", "R", "ONE"),
                    ("T", "R", "R"), ("MF", "C"), ("THREE",), ("MF",), ("H",),
                    ("T", "R", "H"), ("T", "R", "ONE"), ("T", "R", "R"),
                    ("T", "H"), ("T", "R"),
                ),
                {
                    "T": (1, 4096),
                    "H": (1024, 7168),
                    "R": (4, 4),
                    "ONE": (1, 1),
                    "MF": (24, 24),
                    "THREE": (3, 3),
                    "C": (4096, 28672),
                },
            ),
        ),
        backend="catlass_dsl",
        source=("catlass_dsl", "Ascend950"),
        smoke_signature=InputSignature(
            "mhc.mega_mhc",
            dtypes=("bfloat16", "bfloat16", "float32", "float32", "float32",
                    "float32", "float32", "float32", "bfloat16", "bfloat16",
                    "float32", "float32", "bfloat16", "float32"),
            shapes=((96, 5120), (96, 4, 5120), (96, 4), (96, 4, 1), (96, 4, 4),
                    (24, 20480), (3,), (24,), (5120,), (96, 4, 5120), (96, 4, 1),
                    (96, 4, 4), (96, 5120), (96, 4)),
            physical_layouts=("ND",) * 14,
            specializations=(("tensor_contract", "supported"),),
        ),
    ),
)
