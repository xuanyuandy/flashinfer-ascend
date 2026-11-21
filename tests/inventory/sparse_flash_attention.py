"""Reviewed arch22 sparse attention capabilities and wheel targets."""

from flashinfer.runtime import CapabilityCase, CapabilitySpec
from tests.inventory import ReviewedKernel


KERNELS = tuple(
    ReviewedKernel(
        "ascendc/attention/sparse_flash_attention/native/" + soc,
        CapabilitySpec(
            op="attention.sparse_flash_attention",
            soc_models=(soc,),
            dtypes=("float16", "bfloat16", "int32", "float32", "none"),
            cases=tuple(CapabilityCase(
                name="arch22_mla_lse" if lse else "arch22_mla_no_lse",
                compute_dtypes=("float16", "bfloat16"),
                accumulator_dtypes=("float32",), attention_variants=("mla",),
                kv_cache_types=("dense", "ragged", "paged"), causal_values=(False, True),
                sparse_formats=("block_sparse",),
                auxiliary_outputs=("softmax_max", "softmax_sum") if lse else (),
            ) for lse in (False, True)),
        ),
        backend="ascendc", native_recipe=("sparse_flash_attention", "native"),
    )
    for soc in ("Ascend910B", "Ascend910C")
)
