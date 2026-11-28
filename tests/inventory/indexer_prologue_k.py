"""Reviewed mutation-only IndexerPrologueK capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/indexer_prologue_k/native/op_package/" + soc,
        CapabilitySpec(
            op="attention.indexer_prologue_k",
            soc_models=(soc,),
            dtypes=("bfloat16", "float16", "int8", "int32", "float32"),
            cases=tuple(
                CapabilityCase(
                    name="mixed_cache_mutation_" + norm,
                    compute_dtypes=("bfloat16",),
                    accumulator_dtypes=("float32",),
                    fusion_ops=("matmul", "rmsnorm", "rope", "dynamic_quantize"),
                )
                for norm in ("bf16", "fp32")
            ),
        ),
        backend="ascendc", native_recipe=("indexer_prologue_k", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
