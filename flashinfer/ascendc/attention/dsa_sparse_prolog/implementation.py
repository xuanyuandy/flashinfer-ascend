"""A2/A3 arch22 DsaSparseProlog. The source op name and ACLNN entry stay unchanged."""

from flashinfer.ascendc._runtime import native_admission
from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import ShapeConstraint, TensorParameter
from flashinfer.operators.attention.dsa_sparse_prolog.operator import OPERATOR
from flashinfer.operators.author_api import Support
from flashinfer.runtime.targets import ARCH22_SOCS
from .runner import make_runner

_FLOAT_LIMIT = 3.4028234663852886e38


def _tensor(name, role, dtype, shape, alternate=(), layouts=("ND",)):
    return TensorParameter(
        name=name, role=role, dtype=dtype, shape=shape,
        shape_alternatives=alternate, physical_layouts=layouts,
    )


def _compute_signature(compute):
    return (compute,) * 8 + (compute,) * 6 + ("float32", compute, "float32")


_LIMITS = (
    ("T", 1048576), ("B", 4096), ("S", 1048576), ("He", 16384),
    ("Hcq", 8192), ("Hckv", 8192), ("Dr", 512), ("N", 256), ("D", 1024),
    ("Wdq0", 32768), ("Wdq1", 32768), ("Wuq0", 32768), ("Wuq1", 32768),
    ("Wkv0", 32768), ("Wkv1", 32768),
    ("NzDq0", 4096), ("NzDq1", 4096), ("NzUq0", 4096), ("NzUq1", 4096),
    ("NzKv0", 4096), ("NzKv1", 4096),
)


IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    default_target="Ascend910B",
    support=Support(
        soc_models=ARCH22_SOCS,
        dtype=("float16", "bfloat16", "float32"),
        physical_layouts=("ND", "NZ"),
        specializations=(("tensor_contract", ("supported",)),),
    ),
    build=BuildConfig(
        device="op_kernel/dsa_sparse_prolog.cpp",
        host=(
            "op_host/dsa_sparse_prolog_def.cpp",
            "op_host/dsa_sparse_prolog_infershape.cpp",
            "op_host/arch22/dsa_sparse_prolog_tiling.cpp",
            "op_host/arch22/dsa_sparse_prolog_tiling_register.cpp",
        ),
        op_type="DsaSparseProlog",
        tiling="TilingDsaSparseProlog",
        cmake="CMakeLists.txt",
        device_entry="dsa_sparse_prolog",
        task_type="aic_aiv",
        sync_policy="cross_core_flags",
        workspace_bytes="dynamic",
    ),
    binding=NativeBinding(
        tensors=(
            _tensor("token_x", "input", "float16", ("T", "He"), (("B", "S", "He"),)),
            _tensor("weight_dq", "input", "float16", ("Wdq0", "Wdq1"),
                    (("NzDq0", "NzDq1", 16, 16),), ("ND", "NZ")),
            _tensor("weight_uq_qr", "input", "float16", ("Wuq0", "Wuq1"),
                    (("NzUq0", "NzUq1", 16, 16),), ("ND", "NZ")),
            _tensor("weight_dkv_kr", "input", "float16", ("Wkv0", "Wkv1"),
                    (("NzKv0", "NzKv1", 16, 16),), ("ND", "NZ")),
            _tensor("rmsnorm_gamma_cq", "input", "float16", ("Hcq",)),
            _tensor("rmsnorm_gamma_ckv", "input", "float16", ("Hckv",)),
            _tensor("rope_sin", "input", "float16", ("T", "Dr"), (("B", "S", "Dr"),)),
            _tensor("rope_cos", "input", "float16", ("T", "Dr"), (("B", "S", "Dr"),)),
            _tensor("q_nope", "output", "float16", ("T", "N", "D"), (("B", "S", "N", "D"),)),
            _tensor("q_pe", "output", "float16", ("T", "N", "Dr"), (("B", "S", "N", "Dr"),)),
            _tensor("ct_kv", "output", "float16", ("T", "Hckv"), (("B", "S", "Hckv"),)),
            _tensor("k_pe", "output", "float16", ("T", 1, "Dr"), (("B", "S", 1, "Dr"),)),
            _tensor("qr", "output", "float16", ("T", "Hcq"), (("B", "S", "Hcq"),)),
            _tensor("q_m", "output", "float16", ("T", "Hcq"), (("B", "S", "Hcq"),)),
            _tensor("q_rstd", "output", "float32", ("T", 1), (("B", "S", 1),)),
            _tensor("ckv_m", "output", "float16", ("T", "Hckv"), (("B", "S", "Hckv"),)),
            _tensor("ckv_rstd", "output", "float32", ("T", 1), (("B", "S", 1),)),
        ),
        dimensions=tuple(ShapeConstraint(name, 1, maximum) for name, maximum in _LIMITS),
        dtype_signatures=(_compute_signature("bfloat16"), _compute_signature("float16")),
        attributes=(
            AttributeParameter("rmsnorm_epsilon_cq", "float64", 1e-05, 0.0, 1.0),
            AttributeParameter("rmsnorm_epsilon_ckv", "float64", 1e-05, 0.0, 1.0),
            AttributeParameter("mla_scale_q_lora", "float64", 0.0, -_FLOAT_LIMIT, _FLOAT_LIMIT),
            AttributeParameter("mla_scale_kv_lora", "float64", 0.0, -_FLOAT_LIMIT, _FLOAT_LIMIT),
            AttributeParameter("qk_nope_head_dim", "int64", 128, 1, 1024),
            AttributeParameter("is_type_demotion", "bool", False),
            AttributeParameter("apply_rope_fusion", "bool", False),
        ),
    ),
    runner_factory=make_runner,
    admission=native_admission,
    # Host workspace is dynamic and capped by stepBatch<=128. 128 MiB covers
    # the declared dimension envelope plus the CANN system workspace.
    workspace_bound=128 * 1024 * 1024,
)

IMPLEMENTATION = IMPLEMENTATIONS[0]
NATIVE = IMPLEMENTATION.native
