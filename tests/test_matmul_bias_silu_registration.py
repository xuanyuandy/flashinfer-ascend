"""Production registration, dispatch bounds and generic tensor-lifetime checks."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from flashinfer.ascendc import (
    MATMUL_BIAS_SILU_KERNEL_ID, ascendc_descriptors, load_contract,
    tensor_contract_specialization,
)
from flashinfer.runtime import (
    Backend, DispatchError, GraphCapability, HardwareSignature, InputSignature,
    dispatch, get_global_registry,
)


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected source read, compiler or runner invocation")


@dataclass
class Metadata:
    shape: tuple
    dtype: str = "bfloat16"
    device: str = "npu:0"
    contiguous: bool = True
    pointer: int = 1024

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer

    def numel(self):
        count = 1
        for value in self.shape:
            count *= value
        return count

    def element_size(self):
        return 2


def arguments(m=32, n=64, k=48):
    return (
        Metadata((m, k), pointer=1024), Metadata((k, n), pointer=100000),
        Metadata((n,), pointer=200000), Metadata((m, n), pointer=300000),
    )


def signature(tensors):
    return InputSignature(
        "gemm.matmul_bias_silu", dtypes=tuple(t.dtype for t in tensors),
        shapes=tuple(t.shape for t in tensors), physical_layouts=("ND",) * len(tensors),
        specializations=(("tensor_contract", tensor_contract_specialization(
            MATMUL_BIAS_SILU_KERNEL_ID, tensors,
        )),),
    )


def test_frozen_production_registration_does_not_probe_toolchain_or_sources(monkeypatch):
    from flashinfer import ascendc
    from flashinfer.jit import AscendCToolchain, recipe

    ascendc._contracts.cache_clear()
    monkeypatch.setattr(AscendCToolchain, "discover", forbidden)
    monkeypatch.setattr(recipe, "_source_bytes", forbidden)
    descriptor, contract, spec = load_contract()
    bound = next(item for item in ascendc_descriptors() if item.kernel_id == descriptor.kernel_id)
    assert bound.runner.provenance is None
    assert bound.fingerprint == descriptor.fingerprint
    assert bound.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert bound.workspace.constant_bytes == (16 << 20) + 4 * 256 * 256
    assert contract.tensor_parameter_names == ("a", "b", "bias", "out")
    assert contract.device_parameter_order == ("a", "b", "bias", "out", "workspace", "tiling")
    assert dict(spec.input_fingerprints)["kernel"] == descriptor.fingerprint
    assert dict(spec.input_fingerprints)["build_recipe"] == contract.fingerprint


def test_ordinary_global_registry_dispatches_first_production_entry(monkeypatch):
    from flashinfer.jit.runtime import AotKernelRunner

    monkeypatch.setattr(AotKernelRunner, "resolve", forbidden)
    selected = dispatch(signature(arguments()), HardwareSignature("Ascend950"))
    assert selected.kernel_id == MATMUL_BIAS_SILU_KERNEL_ID
    descriptor = get_global_registry().get(selected.kernel_id)
    assert descriptor.backend == Backend.ASCENDC
    assert isinstance(descriptor.runner, AotKernelRunner)
    assert descriptor.runner.provenance is None


@pytest.mark.parametrize("m,n,k,diagnostic", [
    (272, 16, 16, "M=272"), (16, 272, 16, "N=272"),
    (16, 16, 272, "K=272"), (17, 16, 16, "multiple_of=16"),
    (16, 24, 16, "multiple_of=16"), (16, 16, 8, "K=8"),
])
def test_declared_range_and_alignment_fail_in_dispatch(monkeypatch, m, n, k, diagnostic):
    from flashinfer.jit.runtime import AotKernelRunner

    monkeypatch.setattr(AotKernelRunner, "resolve", forbidden)
    with pytest.raises(DispatchError, match=diagnostic):
        dispatch(signature(arguments(m, n, k)), HardwareSignature("Ascend950"))


@pytest.mark.parametrize("index,changes,diagnostic", [
    (1, {"shape": (32, 64)}, "inconsistent"),
    (2, {"shape": (1, 64)}, "rank"),
    (3, {"dtype": "float32"}, "dtype for out"),
    (0, {"contiguous": False}, "contiguous"),
    (2, {"device": "npu:1"}, "share one device"),
])
def test_cross_tensor_metadata_fail_in_dispatch(index, changes, diagnostic):
    tensors = list(arguments())
    tensors[index] = replace(tensors[index], **changes)
    with pytest.raises(DispatchError, match=diagnostic):
        dispatch(signature(tuple(tensors)), HardwareSignature("Ascend950"))


@pytest.mark.parametrize("alias", ["same_object", "overlapping_storage", "two_outputs"])
def test_generic_tensor_adapter_rejects_every_output_alias(alias):
    from flashinfer.jit.tensor_adapter import _reject_output_overlap

    parameters = [SimpleNamespace(role="input"), SimpleNamespace(role="output")]
    first = Metadata((16, 16), pointer=1024)
    second = first if alias == "same_object" else Metadata((16, 16), pointer=1280)
    if alias == "two_outputs":
        parameters[0] = SimpleNamespace(role="output")
    with pytest.raises(ValueError, match="must not overlap"):
        _reject_output_overlap(parameters, (first, second))


def test_generic_tensor_adapter_accepts_adjacent_storage():
    from flashinfer.jit.tensor_adapter import _reject_output_overlap

    _reject_output_overlap(
        (SimpleNamespace(role="input"), SimpleNamespace(role="output")),
        (Metadata((16, 16), pointer=1024), Metadata((16, 16), pointer=1536)),
    )


@pytest.mark.parametrize("supply_output", [False, True])
def test_public_entry_dispatches_default_registered_runner(monkeypatch, supply_output):
    import torch
    from flashinfer import gemm

    captured = []
    actual_build_signature = gemm.build_signature

    def host_signature(*args, **kwargs):
        return replace(actual_build_signature(*args, **kwargs), physical_layouts=("ND",) * 4)

    monkeypatch.setattr(gemm, "build_signature", host_signature)
    monkeypatch.setattr(gemm, "current_hardware", lambda: HardwareSignature("Ascend950"))
    monkeypatch.setattr(gemm, "_run_selected", lambda kernel_id, *tensors: captured.append(
        (kernel_id, tensors)))
    tensors = tuple(torch.empty(shape, dtype=torch.bfloat16)
                    for shape in ((16, 32), (32, 48), (48,), (16, 48)))
    output = gemm.matmul_bias_silu(*tensors[:3], out=tensors[3] if supply_output else None)
    if supply_output:
        assert output is tensors[3]
    assert tuple(output.shape) == (16, 48) and output.dtype == torch.bfloat16
    assert captured[0][0] == MATMUL_BIAS_SILU_KERNEL_ID
    assert all(actual is expected for actual, expected in zip(captured[0][1][:3], tensors[:3]))
    assert captured[0][1][3] is output


def test_public_oversized_shape_rejected_before_output_allocation(monkeypatch):
    import torch
    from flashinfer import gemm

    # Metadata-only inputs model an output too large to allocate safely.
    a, b, bias = arguments(1 << 40, 32, 48)[:3]
    for tensor in (a, b, bias):
        tensor.ndim = len(tensor.shape)
    monkeypatch.setattr(gemm, "current_hardware", lambda: HardwareSignature("Ascend950"))
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(gemm, "_run_selected", forbidden)
    with pytest.raises(DispatchError, match="1099511627776"):
        gemm.matmul_bias_silu(a, b, bias)


def test_list_kernels_reports_runtime_aot_without_compiler(monkeypatch, capsys):
    from flashinfer import cli
    from flashinfer.jit import AscendCToolchain, op_package, recipe

    monkeypatch.setattr(AscendCToolchain, "discover", forbidden)
    monkeypatch.setattr(recipe, "_source_bytes", forbidden)
    monkeypatch.setattr(op_package, "load_op_package", forbidden)
    assert cli.main(["list-kernels"]) == 0
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert set(lines) == {
        MATMUL_BIAS_SILU_KERNEL_ID + "\tgemm.matmul_bias_silu\tascendc\taot (runtime discovery)",
        FLASH_ATTENTION_KERNEL_ID + "\tascendc.single_prefill_catlass\tascendc\taot (runtime discovery)",
    }


def test_list_kernels_keeps_reference_location(monkeypatch, capsys):
    from flashinfer import cli
    from flashinfer.runtime import KernelDescriptor

    descriptor = KernelDescriptor("reference/example", "example", Backend.REFERENCE)
    monkeypatch.setattr(cli, "load_kernel_manifest", lambda _: (descriptor,))
    assert cli.main(["list-kernels"]) == 0
    assert capsys.readouterr().out.strip() == "reference/example\texample\treference\treference"
