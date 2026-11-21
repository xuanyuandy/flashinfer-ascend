"""Reviewed arch22 unpacking limits for the three public QKV outputs."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel, shape_ranges


UB_ELEMENTS = 180 * 1024 // 2
AXES = (("Split", "S", "B", "Group", "D"), ("S", "B", "Nq", "D"),
        ("S", "B", "Nkv", "D"), ("S", "B", "Nkv", "D"))
BOUNDS = {"Split": (1, UB_ELEMENTS), "S": (1, 2**31 - 1), "B": (1, 2**31 - 1),
          "Group": (1, UB_ELEMENTS), "D": (16, UB_ELEMENTS),
          "Nq": (1, UB_ELEMENTS), "Nkv": (1, UB_ELEMENTS)}
DTYPES = ("float16", "bfloat16")

KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/unpermute_qkv/native/" + soc,
        CapabilitySpec(op="attention.unpermute_qkv", soc_models=(soc,), dtypes=DTYPES,
                       physical_layouts=("ND",), shape_ranges=shape_ranges(AXES, BOUNDS),
                       cases=(CapabilityCase(name="unpack_heads", compute_dtypes=DTYPES),)),
        backend="ascendc", native_recipe=("unpermute_qkv", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
