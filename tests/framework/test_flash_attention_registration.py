"""Host checks for the explicit example-49 prefill contract and dispatch gate."""

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from flashinfer.ascendc import (
    FLASH_ATTENTION_KERNEL_ID, ascendc_descriptors, load_contract,
    tensor_contract_specialization,
)
from flashinfer.runtime import (
    DispatchError, GraphCapability, HardwareSignature, InputSignature, dispatch,
)


def forbidden(*args, **kwargs):
    raise AssertionError("unsupported input reached allocation, compiler or runner")


@dataclass
class Metadata:
    shape: tuple
    dtype: str = "float16"
    device: str = "npu:0"
    contiguous: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous


def arguments(s=128, t=256, h=2, d=128):
    return tuple(Metadata(shape) for shape in ((s, h, d), (t, h, d), (t, h, d), (s, h, d)))


def signature(tensors):
    return InputSignature(
        "ascendc.single_prefill_catlass", dtypes=tuple(t.dtype for t in tensors),
        shapes=tuple(t.shape for t in tensors), physical_layouts=("ND",) * len(tensors),
        specializations=(("tensor_contract", tensor_contract_specialization(
            FLASH_ATTENTION_KERNEL_ID, tensors,
        )),),
    )


def test_frozen_catlass_declaration_requires_no_dependency_or_compiler(monkeypatch):
    from flashinfer import ascendc
    from flashinfer.jit import AscendCToolchain, recipe

    ascendc._contracts.cache_clear()
    monkeypatch.setattr(AscendCToolchain, "discover", forbidden)
    monkeypatch.setattr(recipe, "_source_bytes", forbidden)
    monkeypatch.setattr(recipe, "_header_tree", forbidden)
    monkeypatch.setenv("FLASHINFER_CATLASS_ROOT", "/mnt/fi/missing-catlass-for-host-test")
    descriptor, contract, spec = load_contract(FLASH_ATTENTION_KERNEL_ID)
    bound = next(item for item in ascendc_descriptors() if item.kernel_id == descriptor.kernel_id)
    assert bound.runner.provenance is None
    assert bound.fingerprint == descriptor.fingerprint
    assert bound.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert bound.workspace.constant_bytes == 16 << 20
    assert contract.version == contract.tiling_schema_version == 2
    assert contract.implementation == "catlass" and contract.output_kind == "op_package"
    assert contract.task_type == "aic_aiv" and contract.sync_policy == "cross_core_flags"
    assert contract.workspace_bytes == "dynamic" and contract.tiling_mode == "host_dynamic"
    assert contract.tiling_host_entry == contract.workspace_host_entry == "FiFlashAttentionTiling"
    assert contract.tensor_parameter_names == ("q", "k", "v", "out")
    assert contract.device_parameter_order == ("q", "k", "v", "out", "workspace", "tiling")
    assert [(item.name, item.revision, item.include_dir) for item in contract.dependencies] == [
        ("catlass", "6c0a3242b2effd861ac91aab1999f2226b643614", "include"),
    ]
    assert dict(spec.input_fingerprints)["kernel"] == descriptor.fingerprint
    assert dict(spec.input_fingerprints)["build_recipe"] == contract.fingerprint


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("s,t,h", [(128, 128, 1), (256, 512, 3), (512, 128, 8), (512, 512, 8)])
def test_supported_envelope_dispatches_without_resolving_runner(monkeypatch, s, t, h, dtype):
    from flashinfer.jit.runtime import AotKernelRunner

    monkeypatch.setattr(AotKernelRunner, "resolve", forbidden)
    tensors = tuple(replace(tensor, dtype=dtype) for tensor in arguments(s, t, h))
    result = dispatch(signature(tensors), HardwareSignature("Ascend950"))
    assert result.kernel_id == FLASH_ATTENTION_KERNEL_ID


@pytest.mark.parametrize("s,t,h,d,diagnostic", [
    (64, 128, 1, 128, "S=64"), (640, 128, 1, 128, "S=640"),
    (192, 128, 1, 128, "multiple_of=128"), (128, 64, 1, 128, "T=64"),
    (128, 640, 1, 128, "T=640"), (128, 192, 1, 128, "multiple_of=128"),
    (128, 128, 0, 128, "positive integer dimensions"), (128, 128, 9, 128, "H=9"),
    (128, 128, 1, 64, "D=64"), (128, 128, 1, 256, "D=256"),
])
def test_shape_envelope_rejected_in_dispatch(monkeypatch, s, t, h, d, diagnostic):
    from flashinfer.jit.runtime import AotKernelRunner

    monkeypatch.setattr(AotKernelRunner, "resolve", forbidden)
    with pytest.raises(DispatchError, match=diagnostic):
        dispatch(signature(arguments(s, t, h, d)), HardwareSignature("Ascend950"))


@pytest.mark.parametrize("index,changes,diagnostic", [
    (0, {"shape": (1, 128, 2, 128)}, "rank"),
    (1, {"shape": (256, 1, 128)}, "inconsistent"),
    (2, {"shape": (128, 2, 128)}, "inconsistent"),
    (3, {"shape": (256, 2, 128)}, "inconsistent"),
    (0, {"dtype": "bfloat16"}, "dtype signature"),
    (3, {"dtype": "float32"}, "dtype for out"),
    (1, {"contiguous": False}, "contiguous"),
    (2, {"device": "npu:1"}, "share one device"),
])
def test_cross_tensor_envelope_rejected_in_dispatch(index, changes, diagnostic):
    tensors = list(arguments())
    tensors[index] = replace(tensors[index], **changes)
    with pytest.raises(DispatchError, match=diagnostic):
        dispatch(signature(tensors), HardwareSignature("Ascend950"))


def _mock_npu_dispatch(monkeypatch):
    from flashinfer.attention import catlass_prefill

    actual = catlass_prefill.build_signature
    monkeypatch.setattr(catlass_prefill, "build_signature", lambda *a, **k: replace(
        actual(*a, **k), physical_layouts=("ND",) * 4,
    ))
    monkeypatch.setattr(catlass_prefill, "current_hardware", lambda: HardwareSignature("Ascend950"))
    return catlass_prefill


@pytest.mark.parametrize("supply_output", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_torch_device_implementation_uses_selected_package_runner(monkeypatch, supply_output, dtype):
    import torch
    import flashinfer

    entry = _mock_npu_dispatch(monkeypatch)
    calls = []
    monkeypatch.setattr(entry, "_run_selected", lambda *args: calls.append(args))
    tensors = tuple(torch.empty(item.shape, dtype=getattr(torch, dtype)) for item in arguments())
    out = entry._single_prefill_impl(
        *tensors[:3], out=tensors[3] if supply_output else None,
    )
    assert out.shape == tensors[0].shape and out.dtype == tensors[0].dtype
    assert len(calls) == 1 and calls[0][0] == FLASH_ATTENTION_KERNEL_ID
    assert all(actual is expected for actual, expected in zip(calls[0][1:4], tensors[:3]))
    assert calls[0][4] is out
    if supply_output:
        assert out is tensors[3]


@pytest.mark.parametrize("options,diagnostic", [
    ({"causal": True}, "causal=False"),
    ({"kv_layout": "HND"}, "kv_layout='NHD'"),
    ({"custom_mask": object()}, "custom_mask"),
])
def test_unsupported_options_rejected_before_allocation_or_runner(monkeypatch, options, diagnostic):
    import torch

    entry = _mock_npu_dispatch(monkeypatch)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(entry, "_run_selected", forbidden)
    with pytest.raises(DispatchError, match=diagnostic):
        entry._single_prefill_impl(*arguments()[:3], **options)


def test_oversized_input_rejected_before_output_allocation(monkeypatch):
    import torch

    entry = _mock_npu_dispatch(monkeypatch)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(entry, "_run_selected", forbidden)
    with pytest.raises(DispatchError, match="1099511627776"):
        entry._single_prefill_impl(*arguments(s=1 << 40)[:3])


def test_output_alias_rejected_before_lazy_runner(monkeypatch):
    import torch
    from flashinfer.jit.runtime import AotKernelRunner

    entry = _mock_npu_dispatch(monkeypatch)
    tensors = tuple(torch.empty(item.shape, dtype=torch.float16) for item in arguments()[:3])
    monkeypatch.setattr(entry, "_run_selected", forbidden)
    monkeypatch.setattr(AotKernelRunner, "resolve", forbidden)
    with pytest.raises(DispatchError, match="must not overlap"):
        entry._single_prefill_impl(*tensors, out=tensors[0])


def test_missing_external_dependency_is_explicit_and_native_is_unaffected(monkeypatch, tmp_path):
    from flashinfer.ascendc import _dependency_roots
    from flashinfer.runtime import SchemaError

    missing = tmp_path / "unavailable"
    monkeypatch.setenv("FLASHINFER_CATLASS_ROOT", str(missing))
    _, catlass, _ = load_contract(FLASH_ATTENTION_KERNEL_ID)
    _, native, _ = load_contract()
    assert _dependency_roots(native, Path(__file__).parents[2]) == {}
    with pytest.raises(SchemaError, match="catlass dependency headers are missing"):
        _dependency_roots(catlass, Path(__file__).parents[2])
