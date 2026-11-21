"""Reviewed physical targets and per-axis limits for QKV head permutation."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel, shape_ranges


AXES = (("S", "B", "Nq", "D"), ("S", "B", "Nkv", "D"),
        ("S", "B", "Nkv", "D"), ("Nout", "S", "B", "D"))
BOUNDS = {"S": (1, 2**31 - 1), "B": (1, 2**31 - 1), "Nq": (1, 4095),
          "Nkv": (1, 4095), "Nout": (3, 12285), "D": (16, 92160)}
DTYPES = ("float16", "bfloat16")

KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/permute_qkv/native/" + soc,
        CapabilitySpec(op="attention.permute_qkv", soc_models=(soc,), dtypes=DTYPES,
                       physical_layouts=("ND",), shape_ranges=shape_ranges(AXES, BOUNDS),
                       cases=(CapabilityCase(name="head_permutation", compute_dtypes=DTYPES),)),
        backend="ascendc", native_recipe=("permute_qkv", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
