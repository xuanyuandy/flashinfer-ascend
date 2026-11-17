"""Import, metadata, routing and explicit storage/attribute contract checks."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer.operators.catalog import definition
from flashinfer.operators import runtime as operator_runtime
from flashinfer.ascendc import load_contract

gmm = definition("grouped_matmul_swiglu_quant")
KERNEL_ID = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B"
from flashinfer.ascendc.gemm.grouped_matmul_swiglu_quant.implementation import NATIVE as native
from flashinfer.jit.aclnn_contract import AttributeParameter, TensorDtypeConversion
from flashinfer.jit.recipe import BuildRecipe
from flashinfer.runtime import (
    Backend, CapabilityCase, GraphCapability, HardwareSignature, InferenceSpec,
    InputSignature, QuantSpec, SchemaError, dispatch, get_global_registry,
)


def inputs(weight_dtype="int8", scale_dtype="float32", groups=1, m=8, k=256, n=256):
    def tensor(shape, dtype):
        return SimpleNamespace(shape=shape, dtype=dtype, device="meta", is_contiguous=lambda: True)
    packed = weight_dtype == "int32"
    x = tensor((m, k), "int8")
    w = tensor((3, k, n // (8 if packed else 1)), weight_dtype)
    scale_shape = (3, groups, n) if groups > 1 else (3, n)
    s = tensor(scale_shape, scale_dtype)
    return (x, w, s, tensor((m,), "float32"), tensor((3,), "int64")), (
        {"bias": tensor((3, n), "float32")} if packed else {})


def request(args, options, layout="NZ"):
    tensors, _, inference = gmm.metadata(*args, **options)
    if layout != "NZ":
        quant = list(inference.quantization)
        quant[1] = (1, replace(quant[1][1], physical_layout=layout))
        inference = replace(inference, quantization=tuple(quant))
    return InputSignature(gmm.op, dtypes=tuple("none" if t is None else str(t.dtype).removeprefix("torch.") for t in tensors),
                          shapes=tuple(() if t is None else tuple(t.shape) for t in tensors),
                          specializations=(("tensor_contract", "supported"),), inference=inference)


def test_production_identity_capability_and_aot_only_policy():
    descriptor = get_global_registry().get(KERNEL_ID)
    assert descriptor.op == gmm.op and descriptor.backend == Backend.ASCENDC
    assert descriptor.capability == native.capability
    assert descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert descriptor.runner.jit_backend_factory is None
    _, recipe, spec = load_contract(KERNEL_ID)
    assert not recipe.allow_jit_fallback
    assert recipe.extended_aclnn
    assert len(recipe.tensor_parameters) == 8 and len(recipe.attributes) == 3
    assert spec.target_soc == "Ascend910B"
    assert recipe.fingerprint == BuildRecipe.from_dict(recipe.to_dict()).fingerprint


def test_import_registers_no_torch_compiler_or_npu_provider():
    code = """
import sys
sys.modules['torch'] = None
sys.modules['torch_npu'] = None
import flashinfer
from flashinfer.operators.catalog import definition
KERNEL_ID = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B"
assert callable(flashinfer.grouped_matmul_swiglu_quant)
assert flashinfer.get_global_registry().get(KERNEL_ID).runner.provenance is None
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("weight_dtype,scale_dtype,groups,layout", [
    ("int8", "float32", 1, "NZ"), ("int8", "float16", 1, "NZ"),
    ("int8", "bfloat16", 1, "NZ"), ("int32", "int64", 1, "ND"),
    ("int32", "int64", 1, "NZ"), ("int32", "int64", 2, "NZ"),
    ("int32", "int64", 4, "ND"),
])
def test_complete_quantization_fusion_and_output_cases_select_aot(weight_dtype, scale_dtype, groups, layout):
    args, options = inputs(weight_dtype, scale_dtype, groups)
    signature = request(args, options, layout)
    assert dispatch(signature, HardwareSignature("Ascend910B", 20)).kernel_id == KERNEL_ID
    assert dispatch(signature, HardwareSignature("Ascend910B", 20), backend="ascendc").kernel_id == KERNEL_ID
    assert signature.inference.fusion_ops == ("grouped_matmul", "dequantize", "swiglu", "dynamic_quantize")
    assert signature.inference.auxiliary_outputs == ("output_scale",)
    if groups > 1:
        assert dict(signature.inference.quantization)[1].group_size == (256 // groups,)


def test_fake_metadata_does_not_assume_the_int4_physical_format():
    args, options = inputs("int32", "int64", n=32)
    tensors, _, _ = gmm.metadata(*args, **options)
    assert tensors[-2].shape == (8, 16)
    arguments = gmm.bind(*args, **options)
    assert gmm.infer(arguments, {"weight": "ND"}).outputs[0].shape == (8, 16)
    assert gmm.infer(arguments, {"weight": "NZ"}).outputs[0].shape == (8, 16)
    with pytest.raises(RuntimeError):
        dispatch(request(args, options, layout="NZ"), HardwareSignature("Ascend910B", 20), kernel_id=KERNEL_ID)
    assert dispatch(request(args, options, layout="ND"), HardwareSignature("Ascend910B", 20)).kernel_id == KERNEL_ID


@pytest.mark.parametrize("change", [
    lambda s: replace(s, inference=None),
    lambda s: replace(s, inference=replace(s.inference, fusion_ops=tuple(reversed(("grouped_matmul", "dequantize", "swiglu", "dynamic_quantize"))))),
    lambda s: replace(s, inference=replace(s.inference, auxiliary_outputs=())),
    lambda s: replace(s, inference=replace(s.inference, quantization=())),
    lambda s: replace(s, inference=replace(s.inference, accumulator_dtype="float32")),
    lambda s: replace(s, inference=replace(s.inference, quantization=tuple(
        (i, replace(q, packing_order="high_nibble_first") if i == 1 else q)
        for i, q in s.inference.quantization))),
])
def test_routing_cannot_infer_or_mix_missing_quantization_and_fusion(change):
    args, options = inputs("int32", "int64", 2)
    signature = change(request(args, options))
    with pytest.raises(RuntimeError):
        dispatch(signature, HardwareSignature("Ascend910B", 20), kernel_id=KERNEL_ID)


def test_reject_other_soc_and_forced_backend():
    signature = request(*inputs())
    for hardware, backend in ((HardwareSignature("Ascend950", 32), "ascendc"),
                              (HardwareSignature("Ascend910B", 20), "cann")):
        with pytest.raises(RuntimeError):
            dispatch(signature, hardware, backend=backend)


@pytest.mark.parametrize("variant", ["int8", "int32"])
def test_fake_functional_and_out_never_dispatch_or_read_values(monkeypatch, variant):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    import flashinfer

    monkeypatch.setattr(operator_runtime, "current_hardware", lambda: pytest.fail("fake must not query hardware"))
    with FakeTensorMode():
        args = (torch.empty(8, 256, dtype=torch.int8),
                torch.empty(3, 256, 32 if variant == "int32" else 256, dtype=getattr(torch, variant)),
                torch.empty(3, 256, dtype=torch.int64 if variant == "int32" else torch.float32),
                torch.empty(8, dtype=torch.float32), torch.empty(3, dtype=torch.int64))
        options = dict(bias=torch.empty(3, 256)) if variant == "int32" else {}
        y, scale = flashinfer.grouped_matmul_swiglu_quant(*args, **options)
        assert y.shape == (8, 128) and y.dtype == torch.int8
        assert scale.shape == (8,) and scale.dtype == torch.float32
        supplied = flashinfer.grouped_matmul_swiglu_quant(*args, out=(y, scale), **options)
        assert supplied[0] is y and supplied[1] is scale


def test_generated_torch_schema_preserves_positional_scalar_compatibility():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flashinfer.aclnn._ops import ensure_registered
    ensure_registered()
    with FakeTensorMode():
        x = torch.empty(8, 256, dtype=torch.int8)
        weight = torch.empty(3, 256, 256, dtype=torch.int8)
        scale = torch.empty(3, 256)
        token_scale = torch.empty(8)
        groups = torch.empty(3, dtype=torch.int64)
        y, output_scale = torch.ops.flashinfer.grouped_matmul_swiglu_quant(
            x, weight, scale, token_scale, groups, None, None, 0.01)
        assert y.shape == (8, 128) and output_scale.shape == (8,)


@pytest.mark.parametrize("mutation,message", [
    (lambda a, o: setattr(a[1], "shape", (3, 128, 256)), "dimensions"),
    (lambda a, o: setattr(a[0], "dtype", "float16"), "dtype"),
    (lambda a, o: o.update(offset=a[3]), "offset"),
    (lambda a, o: o.update(bias=a[3]), "bias"),
    (lambda a, o: o.update(swiglu_limit=-1.0), "attribute"),
    (lambda a, o: o.update(swiglu_limit=float("nan")), "attribute"),
    (lambda a, o: setattr(a[4], "shape", (4,)), "prefixes"),
])
def test_bad_metadata_fails_before_any_allocation_or_aot_loading(mutation, message):
    args, options = inputs()
    mutation(args, options)
    with pytest.raises((SchemaError, ValueError), match=message):
        gmm.metadata(*args, **options)


def test_group_size_range_changes_only_the_explicit_operand_group_dimension():
    quant = QuantSpec("symmetric", "int4_packed", "float32", "int32", granularity="group",
                      group_size=(1,), packing_order="low_nibble_first")
    case = CapabilityCase("bounded", quantization=((1, quant),), quantization_group_size_ranges=((1, 1, 128),))
    signature = InputSignature("test", dtypes=("int8", "int32"), inference=InferenceSpec(
        quantization=((1, replace(quant, group_size=(64,))),)))
    assert not case.unsupported_reasons(signature)
    exact = replace(case, quantization_group_size_ranges=())
    assert exact.unsupported_reasons(signature)
    for changed in (replace(quant, group_size=(256,)), replace(quant, group_size=(64,), scale_dtype="float16"),
                    replace(quant, group_size=(64,), packing_order="high_nibble_first")):
        assert case.unsupported_reasons(replace(signature, inference=InferenceSpec(quantization=((1, changed),))))
    assert CapabilityCase.from_dict(case.to_dict()) == case
    with pytest.raises(SchemaError, match="group ranges"):
        replace(case, quantization_group_size_ranges=((1, 2, 128),))


def test_declared_native_representations_and_attributes_remain_strict():
    with pytest.raises(SchemaError):
        TensorDtypeConversion("float32", "int4", 8)
    with pytest.raises(SchemaError):
        AttributeParameter("limited", "float64", 0, 1, 0)
    _, recipe, _ = load_contract(KERNEL_ID)
    assert recipe.native_dtype_signatures[-1][1:3] == ("int4", "uint64")
    with pytest.raises(SchemaError, match="undeclared"):
        recipe.validate_attributes({"ignored": 1})
    args, options = inputs("int32", "int64")
    tensors, _, _ = gmm.metadata(*args, **options)
    shapes = {p.name: None if t is None else tuple(t.shape) for p, t in zip(recipe.tensor_parameters, tensors)}
    dtypes = {p.name: None if t is None else str(t.dtype).removeprefix("torch.")
              for p, t in zip(recipe.tensor_parameters, tensors)}
    shapes["weight_assistance_matrix"] = dtypes["weight_assistance_matrix"] = None
    with pytest.raises(SchemaError, match="required for this dtype"):
        recipe.validate_tensor_metadata(shapes=shapes, dtypes=dtypes)


def test_standard_headers_are_toolchain_identity_inputs(tmp_path):
    from flashinfer.jit.ascendc import _standard_headers_identity
    include = tmp_path / "include"
    include.mkdir()
    header = include / "cstdint"
    header.write_text("original\n")
    identity = _standard_headers_identity((include,))
    header.write_text("changed\n")
    assert _standard_headers_identity((include,)) != identity


def test_selected_kernel_rejects_capture_before_host_reads_or_launch(monkeypatch):
    import torch_npu
    monkeypatch.setattr(torch_npu.npu, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(operator_runtime, "signature", lambda *a, **k: (InputSignature(gmm.op), (SimpleNamespace(device=SimpleNamespace(type="npu")),), None))
    monkeypatch.setattr(operator_runtime, "current_hardware", lambda: HardwareSignature("Ascend910B", 20))
    monkeypatch.setattr(operator_runtime, "dispatch", lambda *a, **k: SimpleNamespace(kernel_id=KERNEL_ID))
    monkeypatch.setattr(operator_runtime, "definition", lambda _: replace(gmm,
        validate_values=lambda _: pytest.fail("capture must not read Tensor values")))
    with pytest.raises(SchemaError, match="NOT_CAPTURABLE"):
        operator_runtime.execute(gmm.name, {})
