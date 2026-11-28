"""Arch22 AOT implementations of the TND block-index expansion.

The kernel is a pure vector-core op migrated from gitcode ops-transformer
``experimental/attention/block_indexer_rearrange`` (Atlas A2/A3). One source
entry serves Ascend910B and Ascend910C; per-SoC manifests are generated.
"""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.author_api import Support
from flashinfer.operators.attention.block_indexer_rearrange.operator import (
    MAX_BLOCK_COUNT,
    MAX_BLOCK_SIZE,
    MAX_HEAD_NUM,
    MAX_TOKENS,
    OPERATOR,
)
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from .runner import admission, make_runner

DTYPE_SIGNATURE = ("int32", "int64", "int64", "int32")


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    default_target="Ascend910B",
    support=Support(
        soc_models=ARCH22_SOCS,
        dtype=("int32", "int64"),
        layout="ND",
        contiguous=True,
        cases=(
            CapabilityCase(
                name="tnd_block_index_rearrange",
                dtype_signatures=(DTYPE_SIGNATURE,),
                compute_dtypes=("int32",),
                causal_values=(True, False),
            ),
        ),
    ),
    build=BuildConfig(
        device="op_kernel/block_indexer_rearrange.cpp",
        host=("op_host/block_indexer_rearrange_def.cpp",),
        op_type="FiBlockIndexerRearrange",
        tiling="BlockIndexerRearrangeTilingFunc",
        includes=("include", "op_kernel", "op_host"),
        cmake="CMakeLists.txt",
        task_type="aiv_only",
        sync_policy="compiler_default",
        device_entry="fi_block_indexer_rearrange",
        host_entry="aclnnFiBlockIndexerRearrange",
        workspace_bytes="dynamic",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter("topk_indices", "input", "int32", ("T1", "N2", "K")),
            TensorParameter("acc_seq_qlen", "input", "int64", ("BQ",)),
            TensorParameter("acc_seq_kvlen", "input", "int64", ("BK",)),
            TensorParameter("topk_output", "output", "int32", ("T1", "N2", "KW")),
        ),
        dimensions=(
            ShapeConstraint("T1", 1, MAX_TOKENS),
            ShapeConstraint("N2", 1, MAX_HEAD_NUM),
            ShapeConstraint("K", 1, MAX_BLOCK_COUNT),
            ShapeConstraint("BQ", 2, 65537),
            ShapeConstraint("BK", 2, 65537),
            ShapeConstraint("KW", 1, MAX_BLOCK_COUNT * MAX_BLOCK_SIZE),
        ),
        dtype_signatures=(DTYPE_SIGNATURE,),
        attributes=(
            # Order must match the OpDef Attr declaration.
            AttributeParameter("causal", "bool", False),
            AttributeParameter("block_size", "int64", 1, 1, MAX_BLOCK_SIZE),
        ),
    ),
    runner_factory=make_runner,
    admission=admission,
    # The kernel only consumes the CANN system workspace; the host tiler
    # reports the exact size at launch time.
    workspace_bound=16 * 1024 * 1024,
)
