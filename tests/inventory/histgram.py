"""Reviewed arch22 expert counts, tensor types and token range."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec, ShapeRange
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/fused_moe/histgram/native/" + soc,
        CapabilitySpec(op="fused_moe.histgram", soc_models=(soc,), dtypes=("int32",),
                       physical_layouts=("ND",),
                       cases=(CapabilityCase(name="expert_counts", compute_dtypes=("int32",)),),
                       shape_ranges=(ShapeRange("T", 0, 0, 1, 2**31 - 1), ShapeRange("E", 1, 0, 1, 2**24))),
        backend="ascendc", native_recipe=("histgram", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
