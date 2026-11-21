"""Reviewed arch22 expert merge limits and mixed data/count signatures."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel, shape_ranges


AXES = (("T1", "H1"), ("T2", "H1"), ("T1", "H2"), ("T2", "H2"), ("E",), ("E",),
        ("T", "H1"), ("T", "H2"), ("E",))
BOUNDS = {"T1": (1, 2**31 - 1), "T2": (1, 2**31 - 1), "T": (2, 2**31 - 1),
          "E": (1, 1024), "H1": (16, 90112), "H2": (16, 90112)}
DTYPES = ("float16", "bfloat16")
SIGNATURES = tuple((d, d, d, d, "int32", "int32", d, d, "int32") for d in DTYPES)

KERNELS = tuple(
    ReviewedKernel(
        "ascendc/fused_moe/merge_chunk2/native/" + soc,
        CapabilitySpec(op="fused_moe.merge_chunk2", soc_models=(soc,), physical_layouts=("ND",),
                       cases=(CapabilityCase(name="expert_merge", compute_dtypes=DTYPES,
                                             dtype_signatures=SIGNATURES),),
                       shape_ranges=shape_ranges(AXES, BOUNDS)),
        backend="ascendc", native_recipe=("merge_chunk2", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
