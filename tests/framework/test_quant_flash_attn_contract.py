"""C4A4 layout and independent metadata ACLNN contracts without an NPU."""

from itertools import product
from types import SimpleNamespace

import pytest

from flashinfer._resources import source_root
from flashinfer.ascendc._catalog import source_contract
from flashinfer.ascendc.attention.quant_flash_attn.implementation import KERNEL_ID
from flashinfer.ascendc.attention.quant_flash_attn.runner import admission
from flashinfer.ascendc.attention.quant_flash_attn_metadata.implementation import KERNEL_ID as METADATA_KERNEL_ID
from flashinfer.jit.recipe import resolve_build_recipe
from flashinfer.operators.attention.quant_flash_attn.operator import E8M0, FP4, KV_LAYOUTS, OP, OPERATOR, infer, kv_shapes
from flashinfer.runtime import HardwareSignature, InputSignature, dispatch


@pytest.fixture(scope="module")
def recipe():
    return source_contract(KERNEL_ID)["recipe"]


def test_metadata_has_an_independent_native_package(recipe):
    metadata = source_contract(METADATA_KERNEL_ID)["recipe"]
    assert metadata.package_vendor == "flashinfer_quant_flash_attn_metadata"
    assert metadata.package_vendor != recipe.package_vendor
    assert metadata.host_entry == "aclnnFiQuantFlashAttnMetadata"
    assert metadata.task_type == "aicpu"
    assert metadata.device_parameter_order == ()
    assert metadata.tensor_parameter_names == ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv", "metadata")
    for native in (recipe, metadata):
        directory = f"csrc/kernels/attention/{native.recipe_id}/"
        assert all(ref.locator.startswith(directory) for ref in native.sources + native.host_sources)
    shared = "csrc/kernels/attention/common/op_kernel"
    main_headers = {ref.locator for ref in recipe.headers
                    if not ref.locator.startswith("csrc/kernels/attention/quant_flash_attn/")}
    metadata_headers = {ref.locator for ref in metadata.headers
                        if not ref.locator.startswith("csrc/kernels/attention/quant_flash_attn_metadata/")}
    root = source_root()
    all_shared = {path.relative_to(root).as_posix() for path in (root / shared).rglob("*.h")}
    aicpu_headers = {f"{shared}/aicpu_common.h"} | {
        path.relative_to(root).as_posix() for path in (root / shared / "load_balance").rglob("*.h")
    }
    assert metadata_headers == aicpu_headers
    assert main_headers == all_shared - aicpu_headers
    assert f"{shared}/memcopy/parser.h" in main_headers
    assert recipe.native_dtype_signatures[0][:6] == ("float4_e2m1",) * 3 + ("float8_e8m0",) * 3


def test_metadata_build_snapshot_includes_common_aicpu_rules():
    metadata = source_contract(METADATA_KERNEL_ID)["recipe"]
    resolved = resolve_build_recipe(metadata, source_root=source_root(), toolchain_fingerprint="a" * 64)
    build_files = dict(resolved.build_files)
    assert "csrc/cmake/operator/aicpu/CMakeLists.txt" in build_files
    assert metadata.cmake_entry in build_files
    assert (
        "csrc/kernels/attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.json"
        in build_files
    )


def test_public_torch_functional_and_out_schemas_bind():
    for out in (False, True):
        _, signature = OPERATOR.torch_schema(out=out)
        assert signature.parameters["quant_mode"].default == 5


def _arguments(dim=128, layout="PA_BBND", qscale="TND", quant_schema=0, lse=False,
               *, batch=2, heads=6, kv_heads=2):
    def tensor(shape, dtype):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device="npu:0")

    tokens = 7
    kv, ks, vs = kv_shapes(layout, 5, 128, kv_heads, dim)
    qs = (tokens, heads, dim // 64, 2) if qscale == "TND" else (
        kv_heads, tokens, heads // kv_heads, dim // 64, 2)
    return dict(
        q=tensor((tokens, heads, dim // 2), FP4), k=tensor(kv, FP4), v=tensor(kv, FP4),
        q_descale=tensor(qs, E8M0), k_descale=tensor(ks, E8M0), v_descale=tensor(vs, E8M0),
        block_table=tensor((batch, 3), "int32"), cu_seqlens_q=tensor((batch + 1,), "int32"),
        seqused_q=None, seqused_kv=tensor((batch,), "int32"), attn_mask=None,
        metadata=tensor((2, 4096), "int32"), quant_mode=5, softmax_scale=0.125, mask_mode=0,
        win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="TND",
        layout_q_descale=qscale, layout_kv=layout, layout_out="TND", return_softmax_lse=lse,
        quant_schema=quant_schema,
    )


@pytest.mark.parametrize("dim,layout,qscale,quant_schema,lse", product(
    (128, 256), KV_LAYOUTS, ("TND", "N2TGD"), (0, 1, 2), (False, True),
))
def test_c4a4_shape_inference_matches_native_admission(recipe, dim, layout, qscale, quant_schema, lse):
    invocation = infer(_arguments(dim, layout, qscale, quant_schema, lse), None)
    assert tuple(t.shape for t in invocation.outputs) == ((7, 6, dim), (6, 7) if lse else (0,))
    tensors = invocation.inputs + invocation.outputs
    signature = InputSignature(
        OP, dtypes=tuple(t.dtype if t else "none" for t in tensors),
        shapes=tuple(t.shape if t else () for t in tensors), physical_layouts=("ND",) * len(tensors),
        specializations=(("attribute.return_softmax_lse", str(lse)),), inference=invocation.inference,
    )
    assert admission(recipe, signature) == ()
    assert dispatch(signature, HardwareSignature("Ascend950"), kernel_id=KERNEL_ID).kernel_id == KERNEL_ID


def test_c4a4_accepts_source_upper_bounds():
    output, _ = infer(_arguments(batch=65535, heads=256, kv_heads=4), None).outputs
    assert output.shape == (7, 256, 128)


@pytest.mark.parametrize("shape,diagnostic", [
    (dict(batch=65536), "batch size"),
    (dict(heads=257, kv_heads=257), "256 Q and KV heads"),
    (dict(heads=128, kv_heads=1), "64 Q heads per KV head"),
])
def test_c4a4_rejects_shapes_beyond_source_upper_bounds(shape, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        infer(_arguments(**shape), None)
