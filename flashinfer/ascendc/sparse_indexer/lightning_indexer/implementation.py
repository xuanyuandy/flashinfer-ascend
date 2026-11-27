"""Shared A2/A3 arch22 source and runner for lightning_indexer; per-SoC manifests are generated."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.author_api import Support
from flashinfer.operators.sparse_indexer.lightning_indexer.operator import (
    INT64_MAX,
    MAX_INIT_NUM,
    MAX_LOCAL_NUM,
    MAX_TOPK,
    OPERATOR,
)
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from .runner import admission, make_runner

HEAD_DIM = 128


def _tensor(name, role, dtype, shape, alternate=(), optional=False):
    return TensorParameter(name=name, role=role, dtype=dtype, shape=shape,
                           shape_alternatives=alternate, optional=optional)


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    default_target="Ascend910B",
    support=Support(
        soc_models=ARCH22_SOCS, dtype=("float16", "bfloat16", "int32", "int64", "float32", "none"),
        cases=(
            CapabilityCase(
                name="arch22_lightning_indexer",
                compute_dtypes=("float16", "bfloat16"),
                accumulator_dtypes=("float32",), attention_variants=("lightning_indexer",),
                # Dense (BSND) is deliberately not declared: the source kernel
                # reads cumulative [B+1] lengths while its tiler validates [B]
                # per-sequence arrays for BSND, so no BSND call can be correct.
                kv_cache_types=("ragged", "paged"), causal_values=(True,),
            ),
        ),
    ),
    build=BuildConfig(
        device="op_kernel/lightning_indexer.cpp",
        host=("op_host/lightning_indexer_def.cpp",),
        op_type="FiLightningIndexer", tiling="TilingForLightningIndexer",
        includes=("op_kernel", "op_host"),
    ),
    binding=NativeBinding(
        tensors=(
            # BSND (rank 4) vs TND (rank 3); the rank-4 key shape serves both
            # dense [B,S2,1,D] and paged [PAGES,BLOCK,1,D] storage.
            _tensor("query", "input", "float16", ("B", "S1", "N1", "D"), (("T1", "N1", "D"),)),
            _tensor("key", "input", "float16", ("K0", "K1", 1, "D"), (("T2", 1, "D"),)),
            _tensor("weights", "input", "float32", ("B", "S1", "N1"), (("T1", "N1"),)),
            _tensor("cur_seq_lengths_query", "input", "int64", ("BQ",), (), True),
            _tensor("cur_seq_lengths_key", "input", "int64", ("BQ",), (), True),
            _tensor("block_table", "input", "int32", ("BT", "NB"), (), True),
            _tensor("sparse_indices", "output", "int32", ("B", "S1", 1, "K"), (("T1", 1, "K"),)),
            _tensor("sparse_values", "output", "float16", ("B", "S1", 1, "K"), (("T1", 1, "K"),)),
        ),
        dimensions=tuple(ShapeConstraint(name, minimum, maximum) for name, minimum, maximum in (
            ("B", 1, 65536), ("S1", 1, 1048576), ("N1", 1, 1024), ("T1", 1, 1048576),
            ("K0", 1, 1 << 20), ("K1", 1, 1048576), ("T2", 1, 1048576),
            ("NB", 1, 65536), ("K", 1, MAX_TOPK), ("BQ", 1, 65537), ("BT", 1, 65536),
        )) + (ShapeConstraint("D", HEAD_DIM, HEAD_DIM, HEAD_DIM),),
        dtype_signatures=tuple(
            (dtype, dtype, "float32", "int64", "int64", "int32", "int32", dtype)
            for dtype in ("float16", "bfloat16")
        ),
        attributes=(
            AttributeParameter("layout_query", "string", "BSND"),
            AttributeParameter("layout_key", "string", "PA_BSND"),
            AttributeParameter("sparse_count", "int64", 2048, 1, MAX_TOPK),
            AttributeParameter("block_len", "int64", 1, 1, 16),
            AttributeParameter("q_block_len", "int64", 1, 1, 16),
            AttributeParameter("init_num", "int64", 0, 0, MAX_INIT_NUM),
            AttributeParameter("local_num", "int64", 0, 0, MAX_LOCAL_NUM),
            AttributeParameter("sparse_mode", "int64", 3, 0, 3),
            AttributeParameter("pre_tokens", "int64", INT64_MAX, -INT64_MAX, INT64_MAX),
            AttributeParameter("next_tokens", "int64", INT64_MAX, -INT64_MAX, INT64_MAX),
            AttributeParameter("return_value", "bool", False),
        ),
    ),
    runner_factory=make_runner, admission=admission,
    workspace_bound=128 * 1024 * 1024,
)

IMPLEMENTATION = IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
