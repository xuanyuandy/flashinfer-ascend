"""T6.4：物理排布查询、能力过滤、原因链与历史序列化兼容。"""

from dataclasses import FrozenInstanceError, replace
from enum import IntEnum
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flashinfer.runtime.dispatch import dispatch, explain_dispatch
from flashinfer.runtime.manifest import descriptor_from_dict
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import (
    Backend, CapabilitySpec, CompileMode, DispatchError, KernelDescriptor, SchemaError,
)
from flashinfer.runtime.signature import InputSignature, build_signature


_OP = "attention.decode"
_LIMITED = "ascendc/test/nd_only"
_FALLBACK = "cann/test/unrestricted"


def _signature(formats=()):
    return InputSignature(
        _OP, dtypes=("float16", "float16"), shapes=((2, 8, 128), (2, 8, 128)),
        layouts=("NHD", "NHD"), physical_layouts=formats,
    )


@pytest.fixture
def physical_registry():
    return KernelRegistry((
        KernelDescriptor(
            _LIMITED, backend=Backend.ASCENDC, compile_mode=CompileMode.AOT,
            capability=CapabilitySpec(_OP, physical_layouts=("ND",)),
        ),
        KernelDescriptor(_FALLBACK, backend=Backend.CANN, capability=CapabilitySpec(_OP)),
    ))


def test_physical_layouts_are_canonical_frozen_and_hashable():
    capability = CapabilitySpec(_OP, physical_layouts=["NZ", "ND", "NZ"])
    assert capability.physical_layouts == ("ND", "NZ")
    assert capability == CapabilitySpec(_OP, physical_layouts=("ND", "NZ"))
    assert hash(capability) == hash(CapabilitySpec(_OP, physical_layouts=("ND", "NZ")))
    with pytest.raises(FrozenInstanceError):
        capability.physical_layouts = ()
    with pytest.raises(FrozenInstanceError):
        _signature().physical_layouts = ("ND",)


@pytest.mark.parametrize("value", [None, "ND", 29, ("",), ("nd",), ("FRACTAL_NZ",), ("NCHW",), (None,), (["ND"],)])
def test_capability_rejects_invalid_physical_layouts(value):
    with pytest.raises(SchemaError, match="physical_layouts"):
        CapabilitySpec(_OP, physical_layouts=value)
    with pytest.raises(SchemaError, match="physical_layouts"):
        CapabilitySpec.from_dict({"op": _OP, "physical_layouts": value})


@pytest.mark.parametrize("value", [None, "ND", 29, ("nd",), (None,), (2,), ("FRACTAL_NZ",)])
def test_signature_rejects_invalid_physical_layouts(value):
    with pytest.raises(ValueError, match="physical_layouts"):
        InputSignature(_OP, physical_layouts=value)


def test_default_does_not_restrict_or_change_legacy_payload(npu_hardware):
    capability = CapabilitySpec(_OP)
    assert capability.physical_layouts == ()
    assert "physical_layouts" not in capability.to_dict()
    assert CapabilitySpec.from_dict({"op": _OP}) == capability
    for formats in ((), ("ND", "NZ"), ("NZ", "")):
        assert capability.unsupported_reasons(_signature(formats), npu_hardware) == ()
    descriptor = KernelDescriptor(_FALLBACK, capability=capability)
    assert "capability" not in descriptor.to_dict()
    assert descriptor_from_dict(descriptor.to_dict()).fingerprint == descriptor.fingerprint
    assert "physical_layouts" not in _signature().to_dict()
    assert _signature(("", "")) == _signature()
    assert hash(_signature(("", ""))) == hash(_signature())


class _Format(IntEnum):
    ND = 2
    FRACTAL_NZ = 29


@pytest.mark.parametrize("raw, expected", [
    (2, "ND"), (29, "NZ"), (_Format.ND, "ND"), (_Format.FRACTAL_NZ, "NZ"),
    (0, ""), (3, ""), (4, ""), (50, ""), (-1, ""),
])
def test_build_signature_queries_verified_api_without_converting(raw, expected):
    tensor = SimpleNamespace(dtype="float16", shape=(32, 32), device=SimpleNamespace(type="npu"))
    query = Mock(return_value=raw)
    with patch.dict("sys.modules", torch_npu=SimpleNamespace(get_npu_format=query)):
        signature = build_signature(_OP, (tensor,), layouts=("NHD",))
    query.assert_called_once_with(tensor)
    assert signature.physical_layouts == ((expected,) if expected else ())
    assert signature.layouts == ("NHD",)
    assert signature.shapes == ((32, 32),)


def test_query_keeps_tensor_order_and_unknown_entries():
    tensors = tuple(
        SimpleNamespace(dtype="float16", shape=(32, 32), device=SimpleNamespace(type=kind))
        for kind in ("cpu", "npu", "privateuseone", "npu")
    )
    query = Mock(side_effect=(29, 2, RuntimeError("format unavailable")))
    with patch.dict("sys.modules", torch_npu=SimpleNamespace(get_npu_format=query)):
        signature = build_signature(_OP, tensors)
    assert signature.physical_layouts == ("", "NZ", "ND", "")
    assert signature.to_dict()["physical_layouts"] == ["", "NZ", "ND", ""]
    assert query.call_count == 3


@pytest.mark.parametrize("device", [None, SimpleNamespace(type="cpu"), SimpleNamespace(type="npu")])
def test_missing_torch_npu_keeps_signature_constructible(device):
    tensor = SimpleNamespace(dtype="float16", shape=(32, 32), device=device)
    with patch.dict("sys.modules", torch_npu=None):
        assert build_signature(_OP, (tensor,)).physical_layouts == ()


@pytest.mark.parametrize("allowed, actual, accepted", [
    (("ND",), ("ND", "ND"), True),
    (("ND",), ("ND", "NZ"), False),
    (("NZ",), ("NZ", "NZ"), True),
    (("ND", "NZ"), ("ND", "NZ"), True),
    (("ND", "NZ"), ("ND", ""), False),
    (("ND",), ("ND",), False),
    (("ND",), (), False),
])
def test_physical_layout_constraint_applies_to_every_input(npu_hardware, allowed, actual, accepted):
    capability = CapabilitySpec(_OP, layouts=("NHD",), physical_layouts=allowed)
    reasons = capability.unsupported_reasons(_signature(actual), npu_hardware)
    assert (not reasons) == accepted
    if not accepted:
        assert all("物理排布" in reason and "支持集" in reason for reason in reasons)


def test_missing_tensor_metadata_rejects_declared_layout(npu_hardware):
    reasons = CapabilitySpec(_OP, physical_layouts=("ND",)).unsupported_reasons(
        InputSignature(_OP), npu_hardware,
    )
    assert reasons == ("缺少输入张量的物理排布，无法检查支持集 ['ND']",)


def test_dispatch_rejects_mismatch_and_explain_keeps_all_reasons(physical_registry, npu_hardware):
    signature = replace(_signature(("ND", "NZ")), dtypes=("int8", "float16"))
    limited = physical_registry.get(_LIMITED)
    registry = KernelRegistry((
        replace(limited, capability=replace(limited.capability, dtypes=("float16",))),
        physical_registry.get(_FALLBACK),
    ))
    decision = dispatch(signature, npu_hardware, registry=registry)
    assert decision.kernel_id == _FALLBACK
    report = next(r for r in decision.reports if r.kernel_id == _LIMITED)
    assert not report.accepted
    assert report.reasons == (
        "dtype int8 不在支持集 ['float16']",
        "输入张量 1 物理排布 NZ 不在支持集 ['ND']",
    )
    explanation = explain_dispatch(signature, npu_hardware, registry=registry)
    assert f"[reject] {_LIMITED}" in explanation
    assert f"[accept] {_FALLBACK}" in explanation
    assert all(reason in explanation for reason in report.reasons)


def test_explicit_kernel_and_no_survivors_explain_layout_failure(physical_registry, npu_hardware):
    assert dispatch(_signature(("ND", "ND")), npu_hardware, kernel_id=_LIMITED, registry=physical_registry).kernel_id == _LIMITED
    with pytest.raises(DispatchError, match="输入张量 1 物理排布 NZ"):
        dispatch(_signature(("ND", "NZ")), npu_hardware, kernel_id=_LIMITED, registry=physical_registry)
    with pytest.raises(DispatchError, match="物理排布未知"):
        explain_dispatch(_signature(), npu_hardware, backend="ascendc", registry=physical_registry)


def test_physical_layout_changes_partition_dispatch_cache(physical_registry, npu_hardware):
    nd = _signature(("ND", "ND"))
    nz = _signature(("ND", "NZ"))
    first = dispatch(nd, npu_hardware, registry=physical_registry)
    second = dispatch(nz, npu_hardware, registry=physical_registry)
    unknown = dispatch(_signature(), npu_hardware, registry=physical_registry)
    again = dispatch(nd, npu_hardware, registry=physical_registry)
    assert first.kernel_id == again.kernel_id == _LIMITED
    assert second.kernel_id == unknown.kernel_id == _FALLBACK
    assert not first.cache_hit and not second.cache_hit and not unknown.cache_hit
    assert again.cache_hit


def test_physical_layouts_round_trip_and_affect_fingerprints(physical_registry):
    descriptor = physical_registry.get(_LIMITED)
    payload = json.loads(json.dumps(descriptor.to_dict()))
    assert payload["capability"]["physical_layouts"] == ["ND"]
    assert "shape_ranges" not in payload["capability"]
    restored = descriptor_from_dict(payload)
    assert restored == descriptor
    assert restored.fingerprint == descriptor.fingerprint
    for layouts in ((), ("NZ",), ("ND", "NZ")):
        capability = replace(descriptor.capability, physical_layouts=layouts)
        assert capability.fingerprint != descriptor.capability.fingerprint
        assert replace(descriptor, capability=capability).fingerprint != descriptor.fingerprint


def test_physical_apis_are_tracked_and_survive_ledger_refresh():
    from flashinfer.parity import ParityEntry, load_packaged_manifest
    from tools.update_parity_inventory import annotate

    class UpstreamWithoutRuntimeContracts:
        def resolve(self, symbol):
            return None

    entries = {entry.local: entry for entry in load_packaged_manifest("all").entries}
    for symbol in (
        "flashinfer.runtime.schema.CapabilitySpec",
        "flashinfer.runtime.hardware.HardwareSignature",
        "flashinfer.runtime.signature.InputSignature",
        "flashinfer.runtime.signature.build_signature",
    ):
        entry = entries[symbol]
        assert entry.api_kind == "npu_extension"
        assert entry.implementation_status == "framework"
        assert ParityEntry.from_dict(annotate(entry.to_dict(), UpstreamWithoutRuntimeContracts())) == entry
