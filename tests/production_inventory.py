"""Release expectations independent of operator discovery and generated metadata.

The original capability snapshot and its reviewed amendments are preserved here.
New implementations add KERNELS in tests/inventory/<operator>.py.
"""

from collections import Counter, defaultdict
from importlib import import_module
import json
from pathlib import Path
from pkgutil import iter_modules

from flashinfer.runtime import InputSignature
from tests import inventory


def _baseline_capabilities():
    path = Path(__file__).parent / "framework/data/production_capabilities.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(baseline)
    expected["ascendc/t4/flash_attention/catlass/op_package/Ascend910B"] = dict(
        baseline["ascendc/t4/flash_attention/catlass/op_package/Ascend950"],
        soc_models=["Ascend910B"],
    )
    # Only these reviewed general-purpose backends extend their SoC lists.
    # The frozen fixture remains unchanged; every other capability field is exact.
    for kernel_id in (
        'cann/cascade/merge_state_attention_update_v1',
        'cann/decode/cudnn_batch_decode_fia_v1',
        'cann/fused_moe/cutlass_fused_moe_gmm_v1',
        'cann/gemm/bmm_bf16_v1',
        'cann/gemm/mm_bf16_v1',
        'cann/gemm/segment_gemm_v1',
        'cann/norm/fused_add_rmsnorm_aclnn_v1',
        'cann/norm/gemma_rmsnorm_aclnn_v1',
        'cann/norm/layernorm_aclnn_v1',
        'cann/norm/rmsnorm_aclnn_v1',
        'cann/rope/apply_rope_cos_sin_cache_aclnn_v1',
        'cann/rope/apply_rope_cos_sin_cache_out_v1',
        'cann/rope/apply_rope_inplace_v1',
        'cann/rope/apply_rope_pos_ids_inplace_v1',
        'cann/rope/apply_rope_pos_ids_v1',
        'cann/rope/apply_rope_v1',
        'triton/norm/gemma_fused_add_rmsnorm_v1',
    ):
        expected[kernel_id] = dict(baseline[kernel_id],
                                   soc_models=["Ascend910B", "Ascend910C", "Ascend950"])
    for kernel_id in (
        "ascendc/t4/flash_attention/catlass/op_package/Ascend910B",
        "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B",
    ):
        expected[kernel_id.rsplit("/", 1)[0] + "/Ascend910C"] = dict(
            expected[kernel_id], soc_models=["Ascend910C"])
    return expected


_BASE_NATIVE = (
    ("ascendc/t4/flash_attention/catlass/op_package/Ascend910C", "ascendc.single_prefill_catlass", "flash_attention", "catlass", "Ascend910C"),
    ("ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910C", "gemm.grouped_matmul_swiglu_quant", "grouped_matmul_swiglu_quant", "native", "Ascend910C"),
    ("ascendc/t2/matmul_bias_silu/native/op_package/Ascend950", "gemm.matmul_bias_silu", "matmul_bias_silu", "native", "Ascend950"),
    ("ascendc/t4/flash_attention/catlass/op_package/Ascend910B", "ascendc.single_prefill_catlass", "flash_attention", "catlass", "Ascend910B"),
    ("ascendc/t4/flash_attention/catlass/op_package/Ascend950", "ascendc.single_prefill_catlass", "flash_attention", "catlass", "Ascend950"),
    ("ascendc/gemm/matmul/native/op_package/Ascend950", "gemm.matmul", "matmul", "native", "Ascend950"),
    ("ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B", "gemm.grouped_matmul_swiglu_quant", "grouped_matmul_swiglu_quant", "native", "Ascend910B"),
)


def _reviewed_kernels():
    for module in iter_modules(inventory.__path__, inventory.__name__ + "."):
        yield from import_module(module.name).KERNELS


REVIEWED_CAPABILITIES = _baseline_capabilities()
REVIEWED_BACKENDS = {key: key.split("/", 1)[0] for key in REVIEWED_CAPABILITIES}
# Both dense-prefill implementations have the same reviewed input envelope.
REVIEWED_EXPERIMENTAL = {
    "triton/attention/dense_prefill_v1/Ascend950": (
        "triton", REVIEWED_CAPABILITIES["ascendc/t4/flash_attention/catlass/op_package/Ascend950"],
    ),
}
_native = list(_BASE_NATIVE)
SOURCES = {"triton/gemm/matmul_v1/Ascend950": ("triton", "Ascend950")}
_matmul_sample = InputSignature(
    "gemm.matmul", dtypes=("bfloat16",) * 3, shapes=((16, 16),) * 3,
    physical_layouts=("ND",) * 3, specializations=(("tensor_contract", "supported"),),
)
SMOKE_SIGNATURES = {key: _matmul_sample for key, cap in REVIEWED_CAPABILITIES.items()
                    if cap["op"] == _matmul_sample.op}

for record in _reviewed_kernels():
    key = record.kernel_id
    if key in REVIEWED_CAPABILITIES or key in REVIEWED_EXPERIMENTAL:
        raise ValueError("duplicate reviewed kernel: " + key)
    if record.scope == "experimental":
        REVIEWED_EXPERIMENTAL[key] = (record.backend, record.capability.to_dict())
        continue
    if record.scope != "global":
        raise ValueError("unknown review scope for " + key)
    REVIEWED_CAPABILITIES[key] = record.capability.to_dict()
    REVIEWED_BACKENDS[key] = record.backend
    if record.native_recipe is not None:
        soc, = record.capability.soc_models
        _native.append((key, record.capability.op, *record.native_recipe, soc))
    if record.source is not None:
        SOURCES[key] = record.source
    if record.smoke_signature is not None:
        SMOKE_SIGNATURES[key] = record.smoke_signature

PRODUCTION = tuple(_native)
PRODUCTION_KERNEL_IDS = frozenset(row[0] for row in PRODUCTION)
PRODUCTION_OPS = frozenset(row[1] for row in PRODUCTION)
LEGACY_KERNEL_IDS = frozenset(REVIEWED_CAPABILITIES) - PRODUCTION_KERNEL_IDS
DEFAULT_DESCRIPTOR_COUNT = len(REVIEWED_CAPABILITIES)
LEGACY_DESCRIPTOR_COUNT = len(LEGACY_KERNEL_IDS)
STACK_KERNEL_IDS = defaultdict(set)
for key, capability in REVIEWED_CAPABILITIES.items():
    STACK_KERNEL_IDS[(REVIEWED_BACKENDS[key], capability["op"])].add(key)
STACK_COUNTS = Counter(REVIEWED_BACKENDS.values())


def native_kernel_ids(*socs, native_recipe=None):
    return {kernel for kernel, _, recipe, implementation, soc in PRODUCTION
            if soc in socs and (native_recipe is None or (recipe, implementation) == native_recipe)}


PUBLIC_RECIPES = frozenset((r[2], r[3], r[4], 2, "op_package") for r in PRODUCTION) | frozenset({
    ("matmul_probe", "native", "Ascend950", 2, "op_package"),
    ("matmul_probe", "catlass", "Ascend950", 2, "op_package"),
    ("matmul_probe", "catlass", "Ascend910B", 2, "op_package"),
    ("matmul_probe", "catlass", "Ascend910C", 2, "op_package"),
    ("matmul_probe", "native", "Ascend910B", 2, "op_package"),
    ("matmul_probe", "native", "Ascend910C", 2, "op_package"),
    ("matmul_probe", "native", "ascend910b1", 2, "op_package"),
    ("matmul_probe", "native", "ascend910_9391", 2, "op_package"),
    ("matmul_probe", "native", "ascend310p1", 2, "op_package"),
})
