"""Reviewed dense IndexerPrologueQW capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


DENSE_KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/indexer_prologue_qw/native/op_package/" + soc,
        CapabilitySpec(
            op="attention.indexer_prologue_qw",
            soc_models=(soc,),
            dtypes=("bfloat16", "int8", "float16", "float32"),
            cases=(CapabilityCase(
                name="dense_w8a8_mixed",
                compute_dtypes=("int8",),
                accumulator_dtypes=("int32",),
                fusion_ops=("dynamic_quantize", "quantized_matmul", "rope", "dynamic_quantize"),
                auxiliary_outputs=("q_scale", "w"),
                specializations=(("quant.activation", ("per_token_int8",)),
                                 ("quant.weight", ("per_channel_int8",))),
            ),),
        ),
        backend="ascendc", native_recipe=("indexer_prologue_qw", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)


PREQUANTIZED_KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/indexer_prologue_qw/prequantized/native/op_package/" + soc,
        CapabilitySpec(
            op="attention.indexer_prologue_qw",
            soc_models=(soc,),
            dtypes=("bfloat16", "int8", "float16", "float32"),
            cases=(CapabilityCase(
                name="caller_prequantized_nz_mixed",
                compute_dtypes=("int8",),
                accumulator_dtypes=("int32",),
                fusion_ops=("quantized_matmul", "rope", "dynamic_quantize"),
                auxiliary_outputs=("q_scale", "w"),
                specializations=(("quant.weight", ("caller_prequantized_nz",)),),
            ),),
        ),
        backend="ascendc", native_recipe=("indexer_prologue_qw_prequantized", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)

KERNELS = DENSE_KERNELS + PREQUANTIZED_KERNELS
