"""Rich mutation-only IndexerPrologueK Ascend C declaration."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.attention.indexer_prologue_k.operator import OPERATOR
from flashinfer.operators.author_api import Support
from flashinfer.runtime import CapabilityCase
from flashinfer.runtime.targets import ARCH22_SOCS
from .runner import admission, make_runner

MAX_WORKSPACE_BYTES = 128 * 1024 * 1024


def _cache(name, role, dtype, shape, alias_of=None):
    # Caller-owned caches use independent block/slot strides in the public
    # contract; the upstream binding validates the declared ND strided span.
    return TensorParameter(name=name, role=role, dtype=dtype, shape=shape,
                           allow_strided=True, alias_of=alias_of)


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    name="native/op_package",
    default_target="Ascend910B",
    support=Support(
        soc_models=ARCH22_SOCS,
        dtype=("bfloat16", "float16", "int8", "int32", "float32"),
        cases=tuple(
            CapabilityCase(
                name="mixed_cache_mutation_%s" % norm,
                compute_dtypes=("bfloat16",),
                accumulator_dtypes=("float32",),
                fusion_ops=("matmul", "rmsnorm", "rope", "dynamic_quantize"),
            )
            for norm in ("bf16", "fp32")
        ),
    ),
    build=BuildConfig(
        device="op_kernel/indexer_prologue_k.cpp",
        host=("op_host/indexer_prologue_k_def.cpp",),
        op_type="FiIndexerPrologueK",
        tiling="TilingForIndexerPrologueKFullFusion",
        cmake="CMakeLists.txt",
        device_entry="fi_indexer_prologue_k",
        host_entry="aclnnFiIndexerPrologueK",
        launch_abi="aclnn.indexer_prologue_k.v2",
        package_vendor="flashinfer_indexer_prologue_k",
        recipe_id="indexer_prologue_k",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter(name="latent", role="input", dtype="bfloat16", shape=("T", 512)),
            TensorParameter(name="wk", role="input", dtype="bfloat16", shape=(128, 512)),
            TensorParameter(name="norm_weight", role="input", dtype="bfloat16", shape=(128,)),
            TensorParameter(name="rope_cos", role="input", dtype="float32", shape=("T", 64)),
            TensorParameter(name="rope_sin", role="input", dtype="float32", shape=("T", 64)),
            TensorParameter(name="indices", role="input", dtype="int32", shape=("T", 2)),
            _cache("k_cache", "input", "int8", ("NB", "NS", 1, 128)),
            _cache("k_scale_cache", "input", "float16", ("NB", "NS", 1, 1)),
            _cache("k_cache_out", "output", "int8", ("NB", "NS", 1, 128), "k_cache"),
            _cache("k_scale_cache_out", "output", "float16", ("NB", "NS", 1, 1),
                   "k_scale_cache"),
        ),
        dimensions=(
            ShapeConstraint("T", 1, 1048576),
            ShapeConstraint("NB", 1, 1048576),
            ShapeConstraint("NS", 1, 1048576),
        ),
        dtype_signatures=(
            # Base signature first: OpDef order latent, wk, norm_weight, rope_cos,
            # rope_sin, indices, k_cache, k_scale_cache, k_cache_out, k_scale_cache_out.
            ("bfloat16", "bfloat16", "bfloat16", "float32", "float32", "int32",
             "int8", "float16", "int8", "float16"),
            ("bfloat16", "bfloat16", "float32", "float32", "float32", "int32",
             "int8", "float16", "int8", "float16"),
        ),
        attributes=(
            AttributeParameter("norm_eps", "float64", 1.0e-6, 1.401298464324817e-45,
                               3.4028234663852886e38),
        ),
    ),
    runner_factory=make_runner,
    admission=admission,
    workspace_bound=MAX_WORKSPACE_BYTES,
)

IMPLEMENTATION = IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
