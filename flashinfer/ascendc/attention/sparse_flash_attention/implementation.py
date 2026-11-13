"""A2 arch22 implementation; native build manifests are framework generated."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.author_api import Support
from flashinfer.operators.attention.sparse_flash_attention.operator import INT64_MAX, OPERATOR
from flashinfer.runtime import CapabilityCase
from .runner import admission, make_runner


def _tensor(name, role, dtype, shape, alternate=(), optional=False):
    return TensorParameter(name=name, role=role, dtype=dtype, shape=shape,
                           shape_alternatives=alternate, optional=optional)


IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    support=Support(
        soc="Ascend910B", dtype=("float16", "bfloat16", "int32", "float32", "none"),
        cases=tuple(CapabilityCase(
            name="arch22_mla_lse" if lse else "arch22_mla_no_lse",
            compute_dtypes=("float16", "bfloat16"),
            accumulator_dtypes=("float32",), attention_variants=("mla",),
            kv_cache_types=("dense", "ragged", "paged"), causal_values=(False, True),
            sparse_formats=("block_sparse",),
            auxiliary_outputs=("softmax_max", "softmax_sum") if lse else (),
        ) for lse in (False, True)),
    ),
    build=BuildConfig(
        device="op_kernel/sparse_flash_attention.cpp",
        host=("op_host/sparse_flash_attention_def.cpp",),
        op_type="FiSparseFlashAttention", tiling="TilingSparseFlashAttention",
        cmake="CMakeLists.txt",
    ),
    binding=NativeBinding(
        tensors=(
            _tensor("query", "input", "float16", ("Q0", "Q1", "H", 512), (("Q0", "H", 512),)),
            _tensor("key", "input", "float16", ("K0", "K1", 1, 512), (("K0", 1, 512),)),
            _tensor("value", "input", "float16", ("K0", "K1", 1, 512), (("K0", 1, 512),)),
            _tensor("sparse_indices", "input", "int32", ("Q0", "Q1", 1, "SP"), (("Q0", 1, "SP"),)),
            _tensor("block_table", "input", "int32", ("B", "NB"), optional=True),
            _tensor("actual_seq_lengths_query", "input", "int32", ("B",), optional=True),
            _tensor("actual_seq_lengths_kv", "input", "int32", ("B",), optional=True),
            _tensor("query_rope", "input", "float16", ("Q0", "Q1", "H", 64), (("Q0", "H", 64),), True),
            _tensor("key_rope", "input", "float16", ("K0", "K1", 1, 64), (("K0", 1, 64),), True),
            _tensor("attention_out", "output", "float16", ("Q0", "Q1", "H", 512), (("Q0", "H", 512),)),
            # Disabled auxiliary outputs use two internal one-element buffers.
            # This keeps the native positive-dimension contract unchanged while
            # exposing the upstream (0,) outputs through the public interface.
            _tensor("softmax_max", "output", "float32", (1,), (("Q0", 1, "Q1", "H"), (1, "Q0", "H"))),
            _tensor("softmax_sum", "output", "float32", (1,), (("Q0", 1, "Q1", "H"), (1, "Q0", "H"))),
        ),
        dimensions=tuple(ShapeConstraint(name, 1, maximum) for name, maximum in (
            ("Q0", 1048576), ("Q1", 1048576), ("H", 128), ("K0", 1048576),
            ("K1", 1048576), ("SP", 65536), ("B", 4096), ("NB", 65536),
        )),
        dtype_signatures=tuple(
            (dtype, dtype, dtype, "int32", "int32", "int32", "int32", rope, rope, dtype, "float32", "float32")
            for dtype, rope in (("float16", "float16"), ("bfloat16", "bfloat16"))
        ),
        attributes=(
            AttributeParameter("scale_value", "float64", 1.0, -3.4028234663852886e38, 3.4028234663852886e38),
            AttributeParameter("sparse_block_size", "int64", 1, 1, 128),
            AttributeParameter("layout_query", "string", "BSND"),
            AttributeParameter("layout_kv", "string", "BSND"),
            AttributeParameter("sparse_mode", "int64", 3, 0, 3),
            AttributeParameter("pre_tokens", "int64", INT64_MAX, INT64_MAX, INT64_MAX),
            AttributeParameter("next_tokens", "int64", INT64_MAX, INT64_MAX, INT64_MAX),
            AttributeParameter("attention_mode", "int64", 2, 2, 2),
            AttributeParameter("return_softmax_lse", "bool", False),
        ),
    ),
    runner_factory=make_runner, admission=admission,
    workspace_bound=128 * 1024 * 1024,
)

NATIVE = IMPLEMENTATION.native
