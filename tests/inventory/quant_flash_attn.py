"""Reviewed C4A4 capability and Ascend950 package scope."""

from flashinfer.runtime import CapabilitySpec, InputSignature
from tests.inventory import ReviewedKernel

DTYPES = ("float4_e2m1fn_x2", "float8_e8m0fnu", "int32", "int8", "bfloat16", "float32", "none")
OP = "attention.quant_flash_attn"
KERNELS = (
    ReviewedKernel(
        "ascendc/attention/quant_flash_attn/native/Ascend950",
        CapabilitySpec(OP, soc_models=("Ascend950",), dtypes=DTYPES),
        backend="ascendc",
        native_recipe=("quant_flash_attn", "native"),
        smoke_signature=InputSignature(
            OP,
            dtypes=(DTYPES[0],) * 3 + (DTYPES[1],) * 3
            + ("int32", "none", "int32", "none", "none", "int32", "none", "none",
               "int32", "bfloat16", "float32"),
            shapes=((3, 4, 64), (1, 128, 2, 64), (1, 128, 2, 64), (3, 4, 2, 2),
                    (1, 128, 2, 2, 2), (1, 2, 2, 128, 2), (1, 1), (), (2,), (), (), (1,),
                    (), (), (1, 4096), (3, 4, 128), (4, 3)),
            physical_layouts=("ND",) * 17,
            specializations=(("attribute.return_softmax_lse", "True"),),
        ),
    ),
)
