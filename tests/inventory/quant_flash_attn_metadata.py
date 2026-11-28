"""Reviewed scope of the C4A4 AICPU schedule operator."""

from flashinfer.runtime import CapabilitySpec, InputSignature
from tests.inventory import ReviewedKernel

OP = "attention.quant_flash_attn_metadata"
KERNELS = (
    ReviewedKernel(
        "ascendc/attention/quant_flash_attn_metadata/native/Ascend950",
        CapabilitySpec(OP, soc_models=("Ascend950",), dtypes=("int32", "none")),
        backend="ascendc",
        native_recipe=("quant_flash_attn_metadata", "native"),
        smoke_signature=InputSignature(
            OP, dtypes=("int32", "none", "none", "int32", "int32"),
            shapes=((2,), (), (), (1,), (1, 4096)),
            physical_layouts=("ND", "", "", "ND", "ND"),
        ),
    ),
)
