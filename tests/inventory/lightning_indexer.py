"""Reviewed arch22 lightning indexer capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/sparse_indexer/lightning_indexer/native/" + soc,
        CapabilitySpec(
            op="sparse_indexer.lightning_indexer",
            soc_models=(soc,),
            dtypes=("float16", "bfloat16", "int32", "int64", "float32", "none"),
            cases=(
                CapabilityCase(
                    name="arch22_lightning_indexer",
                    compute_dtypes=("float16", "bfloat16"),
                    accumulator_dtypes=("float32",),
                    attention_variants=("lightning_indexer",),
                    kv_cache_types=("ragged", "paged"),
                    causal_values=(True,),
                ),
            ),
        ),
        backend="ascendc", native_recipe=("lightning_indexer", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
