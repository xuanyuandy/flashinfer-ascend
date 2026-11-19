"""Import, metadata, routing and frozen-contract checks for gemm.grouped_matmul_k_dim."""

from types import SimpleNamespace

import pytest

from flashinfer.ascendc import load_contract
from flashinfer.operators.catalog import definition
from flashinfer.ascendc.gemm.grouped_matmul_k_dim.implementation import NATIVE as native
from flashinfer.jit.recipe import BuildRecipe
from flashinfer.runtime import (
    Backend,
    GraphCapability,
    HardwareSignature,
    InputSignature,
    get_global_registry,
    dispatch,
)

gmm = definition("grouped_matmul_k_dim")
KERNEL_ID = "ascendc/gemm/grouped_matmul_k_dim/native/op_package/Ascend910B"
KERNEL_ID_910C = "ascendc/gemm/grouped_matmul_k_dim/native/op_package/Ascend910C"


def tensor(shape, dtype):
    return SimpleNamespace(shape=shape, dtype=dtype, device="meta", is_contiguous=lambda: True)


def inputs(m=64, n=96, groups=(32, 0, 48, 16), dtype="float16", trans_a=False, fp32_output=False):
    total_k = sum(groups)
    prefixes = []
    running = 0
    for value in groups:
        running += value
        prefixes.append(running)
    x_shape = (total_k, m) if trans_a else (m, total_k)
    return (
        tensor(x_shape, dtype),
        tensor((total_k, n), dtype),
        tensor((len(groups),), "int64"),
    ), dict(trans_a=trans_a, fp32_output=fp32_output)


def request(args, options, layout="ND"):
    tensors, _, inference = gmm.metadata(*args, **options)
    return InputSignature(
        gmm.op,
        dtypes=tuple("none" if t is None else str(t.dtype).removeprefix("torch.") for t in tensors),
        shapes=tuple(() if t is None else tuple(t.shape) for t in tensors),
        specializations=(("tensor_contract", "supported"),),
        physical_layouts=(layout,) * len(tensors),
        inference=inference,
    )


def test_production_identity_and_aot_only_policy():
    descriptor = get_global_registry().get(KERNEL_ID)
    assert descriptor.op == gmm.op and descriptor.backend == Backend.ASCENDC
    assert descriptor.capability == native.capability
    assert descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert descriptor.runner.jit_backend_factory is None
    _, recipe, spec = load_contract(KERNEL_ID)
    assert not recipe.allow_jit_fallback
    assert spec.target_soc == "Ascend910B"
    assert recipe.fingerprint == BuildRecipe.from_dict(recipe.to_dict()).fingerprint


def test_recipe_is_one_device_unit_with_fixed_workspace():
    _, recipe, _ = load_contract(KERNEL_ID)
    assert len(recipe.sources) == 1, "op_package requires exactly one device translation unit"
    assert tuple(p.name for p in recipe.tensor_parameters) == ("x", "weight", "group_list", "y")
    assert tuple(a.name for a in recipe.attributes) == ("trans_a",)
    assert recipe.workspace_bytes == 16 * 1024 * 1024
    assert recipe.workspace_host_entry is None
    assert recipe.tiling_host_entry == "TilingGroupedMatmulKDim"
    assert recipe.device_entry == "fi_grouped_matmul_k_dim"
    assert recipe.host_entry == "aclnnFiGroupedMatmulKDim"


@pytest.mark.parametrize("dtype,fp32_output", [
    ("float16", False), ("float16", True), ("bfloat16", False), ("bfloat16", True),
])
@pytest.mark.parametrize("trans_a", [False, True])
def test_supported_combinations_select_the_arch22_kernel(dtype, fp32_output, trans_a):
    args, options = inputs(dtype=dtype, fp32_output=fp32_output, trans_a=trans_a)
    signature = request(args, options)
    assert dispatch(signature, HardwareSignature("Ascend910B", 20)).kernel_id == KERNEL_ID
    assert dispatch(signature, HardwareSignature("Ascend910C", 20)).kernel_id == KERNEL_ID_910C


def test_metadata_derives_one_tile_per_group():
    args, options = inputs(groups=(32, 0, 48, 16))
    tensors, attributes, inference = gmm.metadata(*args, **options)
    assert tuple(tensors[-1].shape) == (4, 64, 96)
    assert attributes == {"trans_a": False, "fp32_output": False}
    assert str(tensors[-1].dtype).removeprefix("torch.") == "float16"
    assert inference.compute_dtype == "float16"
    assert inference.accumulator_dtype == "float32"


def test_fp32_output_widens_only_the_store():
    args, options = inputs(fp32_output=True)
    tensors, _, inference = gmm.metadata(*args, **options)
    assert str(tensors[-1].dtype).removeprefix("torch.") == "float32"
    assert inference.compute_dtype == "float16"


def test_trans_a_moves_k_between_rows_and_columns():
    untransposed, _ = inputs(trans_a=False)
    transposed, _ = inputs(trans_a=True)
    assert untransposed[0].shape == (64, 96) and transposed[0].shape == (96, 64)
    tensors, _, _ = gmm.metadata(*transposed, trans_a=True, fp32_output=False)
    assert tuple(tensors[-1].shape) == (4, 64, 96)


def test_infer_rejects_k_disagreement():
    args, options = inputs()
    bad = (args[0], tensor((160, 96), "float16"), args[2])
    with pytest.raises(ValueError, match="K extent"):
        gmm.metadata(*bad, **options)


def test_infer_rejects_mixed_input_dtypes():
    args, options = inputs()
    bad = (args[0], tensor((192, 96), "bfloat16"), args[2])
    with pytest.raises(ValueError, match="weight dtype"):
        gmm.metadata(*bad, **options)


def test_infer_rejects_group_list_that_is_not_int64():
    args, options = inputs()
    bad = (args[0], args[1], tensor((4,), "int32"))
    with pytest.raises(ValueError, match="int64"):
        gmm.metadata(*bad, **options)


def test_infer_rejects_stride_above_the_matmul_limit():
    args, options = inputs(m=64, n=70000)
    with pytest.raises(ValueError, match="65535"):
        gmm.metadata(*args, **options)


def test_capability_declares_four_complete_signatures():
    cases = {case.name: case for case in native.capability.cases}
    assert set(cases) == {
        "float16_to_float16", "float16_to_float32",
        "bfloat16_to_bfloat16", "bfloat16_to_float32",
    }
    for name, case in cases.items():
        assert len(case.dtype_signatures) == 1
        signature = case.dtype_signatures[0]
        assert len(signature) == 4 and signature[2] == "int64"
