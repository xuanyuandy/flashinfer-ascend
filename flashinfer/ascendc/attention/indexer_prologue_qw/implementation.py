"""Dense IndexerPrologueQW Ascend C declaration and lazy runner binding."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.attention.indexer_prologue_qw.operator import OPERATOR
from flashinfer.operators.author_api import Support
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from .runner import (
    dense_admission, make_dense_runner, prequantized_admission, make_prequantized_runner,
)

# Native tiling keeps user workspace near 13 MiB at the A2/A3 core counts;
# the bound also covers CANN's system workspace and is checked at build time
# against the observed GetWorkspaceSize result.
MAX_WORKSPACE_BYTES = 128 * 1024 * 1024


def _tensor(name, role, dtype, shape, alternate=()):
    return TensorParameter(name=name, role=role, dtype=dtype, shape=shape,
                           shape_alternatives=alternate)


DENSE_IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    name="native/op_package",
    default_target="Ascend910B",
    support=Support(
        soc_models=ARCH22_SOCS,
        dtype=("bfloat16", "int8", "float16", "float32"),
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
    build=BuildConfig(
        device="op_kernel/indexer_prologue_qw.cpp",
        host=("op_host/indexer_prologue_qw_def.cpp",),
        op_type="FiIndexerPrologueQW",
        tiling="TilingForIndexerPrologueQWFullFusion",
        cmake="CMakeLists.txt",
        device_entry="fi_indexer_prologue_qw",
        host_entry="aclnnFiIndexerPrologueQW",
        launch_abi="aclnn.indexer_prologue_qw.v2",
        package_vendor="flashinfer_indexer_prologue_qw",
        recipe_id="indexer_prologue_qw",
    ),
    binding=NativeBinding(
        tensors=(
            _tensor("qr", "input", "bfloat16", ("T", 1280)),
            _tensor("activation_q", "input", "int8", ("T", 1280)),
            _tensor("weight_q", "input", "int8", (4096, 1280)),
            _tensor("wqb", "input", "bfloat16", (4096, 1280)),
            _tensor("weight_scale", "input", "float32", (4096,)),
            _tensor("activation_scale", "input", "float32", ("T",)),
            _tensor("qb", "input", "bfloat16", ("T", 4096)),
            _tensor("x", "input", "bfloat16", ("T", 5120)),
            _tensor("ww", "input", "bfloat16", (32, 5120)),
            _tensor("wb", "input", "bfloat16", ("T", 32)),
            _tensor("rope_sin", "input", "float32", ("T", 64)),
            _tensor("rope_cos", "input", "float32", ("T", 64)),
            _tensor("q", "output", "int8", ("T", 32, 128)),
            _tensor("q_scale", "output", "float16", ("T", 32)),
            _tensor("w", "output", "float16", ("T", 32)),
        ),
        dimensions=(ShapeConstraint("T", 1, 1048576),),
        attributes=(
            AttributeParameter("w_scale", "float64", 1.0, -3.4028234663852886e38,
                               3.4028234663852886e38),
        ),
    ),
    runner_factory=make_dense_runner,
    admission=dense_admission,
    workspace_bound=MAX_WORKSPACE_BYTES,
)

# --- prequantized QW declaration -------------------------------------------

def _preq_tensor(name, role, dtype, shape, layouts=("ND",)):
    return TensorParameter(name=name, role=role, dtype=dtype, shape=shape,
                           physical_layouts=layouts)


PREQUANTIZED_IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    name="prequantized/native/op_package",
    default_target="Ascend910B",
    support=Support(
        soc_models=ARCH22_SOCS,
        dtype=("bfloat16", "int8", "float16", "float32"),
        cases=(CapabilityCase(
            name="caller_prequantized_nz_mixed",
            compute_dtypes=("int8",),
            accumulator_dtypes=("int32",),
            fusion_ops=("quantized_matmul", "rope", "dynamic_quantize"),
            auxiliary_outputs=("q_scale", "w"),
            specializations=(("quant.weight", ("caller_prequantized_nz",)),),
        ),),
    ),
    build=BuildConfig(
        device="op_kernel/indexer_prologue_qw.cpp",
        host=("op_host/indexer_prologue_qw_def.cpp",),
        op_type="FiIndexerPrologueQWPrequantized",
        tiling="TilingForIndexerPrologueQWFullFusion",
        cmake="CMakeLists.txt",
        device_entry="fi_indexer_prologue_qw_prequantized",
        host_entry="aclnnFiIndexerPrologueQWPrequantized",
        launch_abi="aclnn.indexer_prologue_qw_prequantized.v2",
        package_vendor="flashinfer_indexer_prologue_qw_prequantized",
        recipe_id="indexer_prologue_qw_prequantized",
        definitions=("INDEXER_PROLOGUE_QW_PREQUANTIZED=1",),
    ),
    source_dir="csrc/kernels/attention/indexer_prologue_qw",
    binding=NativeBinding(
        tensors=(
            _preq_tensor("qr", "input", "bfloat16", ("T", 1280)),
            _preq_tensor("activation_q", "input", "int8", ("T", 1280)),
            TensorParameter(name="weight_q", role="input", dtype="int8", shape=(1280, 4096),
                            physical_layouts=("NZ",)),
            TensorParameter(name="wqb", role="input", dtype="int8", shape=(1280, 4096),
                            physical_layouts=("NZ",)),
            _preq_tensor("weight_scale", "input", "float32", (4096,)),
            _preq_tensor("activation_scale", "input", "float32", ("T",)),
            _preq_tensor("qb", "input", "bfloat16", ("T", 4096)),
            _preq_tensor("x", "input", "bfloat16", ("T", 5120)),
            _preq_tensor("ww", "input", "bfloat16", (32, 5120)),
            _preq_tensor("wb", "input", "bfloat16", ("T", 32)),
            _preq_tensor("rope_sin", "input", "float32", ("T", 64)),
            _preq_tensor("rope_cos", "input", "float32", ("T", 64)),
            _preq_tensor("q", "output", "int8", ("T", 32, 128)),
            _preq_tensor("q_scale", "output", "float16", ("T", 32)),
            _preq_tensor("w", "output", "float16", ("T", 32)),
        ),
        dimensions=(ShapeConstraint("T", 1, 1048576),),
        attributes=(
            AttributeParameter("w_scale", "float64", 1.0, -3.4028234663852886e38,
                               3.4028234663852886e38),
        ),
    ),
    runner_factory=make_prequantized_runner,
    admission=prequantized_admission,
    workspace_bound=MAX_WORKSPACE_BYTES,
)

PREQUANTIZED_IMPLEMENTATION = PREQUANTIZED_IMPLEMENTATIONS[0]


IMPLEMENTATIONS = DENSE_IMPLEMENTATIONS + PREQUANTIZED_IMPLEMENTATIONS
IMPLEMENTATION = DENSE_IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
