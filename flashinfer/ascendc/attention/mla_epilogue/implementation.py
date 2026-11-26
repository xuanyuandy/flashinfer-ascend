"""Atlas A3 Ascend C implementation of attention.mla_epilogue."""

from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.recipe import BuildDependency, CATLASS_REVISION, ShapeConstraint, TensorParameter
from flashinfer.operators.author_api import Support
from flashinfer.operators.attention.mla_epilogue.operator import OPERATOR
from flashinfer.runtime import CapabilityCase
from .runner import make_runner


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    default_target="Ascend910C",
    support=Support(
        soc_models=("Ascend910C",),
        dtype=("bfloat16", "float32"),
        cases=(
            CapabilityCase(
                name="a3_mla_epilogue",
                compute_dtypes=("bfloat16",),
                accumulator_dtypes=("float32",),
                attention_variants=("mla",),
            ),
        ),
    ),
    build=BuildConfig(
        device="op_kernel/mla_epilogue.cpp",
        host=("op_host/fi_mla_epilogue.cpp",),
        op_type="FiMlaEpilogue",
        tiling="TilingMlaEpilogue",
        cmake="CMakeLists.txt",
        implementation="catlass",
        dependencies=(BuildDependency(name="catlass", revision=CATLASS_REVISION),),
        definitions=("CATLASS_ARCH=2201",),
        device_entry="fi_mla_epilogue",
    ),
    binding=NativeBinding(
        tensors=(
            TensorParameter(name="x", role="input", dtype="bfloat16", shape=("S", 1, 64, 512)),
            TensorParameter(name="cos", role="input", dtype="float32", shape=("S", 1, 1, 64)),
            TensorParameter(name="sin", role="input", dtype="float32", shape=("S", 1, 1, 64)),
            TensorParameter(name="w1", role="input", dtype="bfloat16", shape=(8, 4096, 1024)),
            TensorParameter(name="w2", role="input", dtype="bfloat16", shape=(5120, 8192)),
            TensorParameter(name="y", role="output", dtype="bfloat16", shape=("S", 5120)),
        ),
        dimensions=(ShapeConstraint(name="S", minimum=1, maximum=131072),),
        dtype_signatures=(
            ("bfloat16", "float32", "float32", "bfloat16", "bfloat16", "bfloat16"),
        ),
    ),
    runner_factory=make_runner,
    # Max workspace includes r/c, system workspace, and the S>=4096 w1/w2 zN/nZ copies.
    workspace_bound=10905190912,
)
IMPLEMENTATION = IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
