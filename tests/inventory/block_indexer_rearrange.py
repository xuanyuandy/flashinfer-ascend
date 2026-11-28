"""Reviewed block_indexer_rearrange capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/block_indexer_rearrange/native/" + soc,
        CapabilitySpec(
            op="attention.block_indexer_rearrange",
            soc_models=(soc,),
            dtypes=("int32", "int64"),
            physical_layouts=("ND",),
            cases=(
                CapabilityCase(
                    name="tnd_block_index_rearrange",
                    dtype_signatures=(("int32", "int64", "int64", "int32"),),
                    compute_dtypes=("int32",),
                    causal_values=(True, False),
                ),
            ),
        ),
        backend="ascendc", native_recipe=("block_indexer_rearrange", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
