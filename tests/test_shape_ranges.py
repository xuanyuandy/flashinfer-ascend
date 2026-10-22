"""T6.3：张量轴规模区间、旧记录兼容性及 dispatch 原因链。"""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from flashinfer.runtime import ShapeRange
from flashinfer.runtime.dispatch import dispatch, explain_dispatch
from flashinfer.runtime.manifest import (
    KERNEL_MANIFEST_SCHEMA_VERSION,
    descriptor_from_dict,
    load_kernel_manifest,
)
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import (
    Backend, CapabilitySpec, CompileMode, DispatchError, KernelDescriptor, SchemaError,
)
from flashinfer.runtime.signature import InputSignature, build_signature


_OP = "attention.decode"
_LIMITED = "ascendc/test/shape_limited"
_FALLBACK = "cann/test/unrestricted"
_RANGES = (
    ShapeRange("head_dim", 0, -1, 64, 128),
    ShapeRange("batch_size", 0, 0, 1, 8),
    ShapeRange("page_size", 1, 1, 16, 32),
    ShapeRange("seq_len", 2, 0, 1, 256),
)


def _signature(shapes=((2, 8, 96), (3, 24, 4, 96), (128, 4, 96))):
    return build_signature(
        _OP, (SimpleNamespace(dtype="torch.float16", shape=shape) for shape in shapes)
    )


@pytest.fixture
def shape_registry():
    return KernelRegistry((
        KernelDescriptor(
            _LIMITED, backend=Backend.ASCENDC, compile_mode=CompileMode.AOT,
            capability=CapabilitySpec(_OP, dtypes=("float16",), shape_ranges=_RANGES),
        ),
        KernelDescriptor(_FALLBACK, backend=Backend.CANN, capability=CapabilitySpec(_OP)),
    ))


def test_shape_ranges_are_frozen_canonical_and_hashable():
    capability = CapabilitySpec(_OP, shape_ranges=list(reversed(_RANGES)))
    assert capability.shape_ranges == _RANGES
    assert capability == CapabilitySpec(_OP, shape_ranges=_RANGES)
    assert hash(capability) == hash(CapabilitySpec(_OP, shape_ranges=_RANGES))
    with pytest.raises(FrozenInstanceError):
        _RANGES[0].minimum = 32
    with pytest.raises(FrozenInstanceError):
        capability.shape_ranges = ()


@pytest.mark.parametrize("changes, message", [
    ({"name": ""}, "name"),
    ({"name": "  "}, "name"),
    ({"name": None}, "name"),
    ({"tensor_index": -1}, "tensor_index"),
    ({"tensor_index": True}, "integer"),
    ({"tensor_index": "0"}, "integer"),
    ({"axis": 1.0}, "integer"),
    ({"axis": False}, "integer"),
    ({"minimum": -1}, "minimum <= maximum"),
    ({"minimum": 129}, "minimum <= maximum"),
    ({"maximum": -1}, "minimum <= maximum"),
    ({"minimum": 64.5}, "integer"),
    ({"maximum": "128"}, "integer"),
    ({"maximum": True}, "integer"),
    ({"maximum": None}, "integer"),
])
def test_invalid_shape_ranges_raise_schema_error(changes, message):
    with pytest.raises(SchemaError, match=message):
        replace(_RANGES[0], **changes)


@pytest.mark.parametrize("ranges", [None, ("head_dim",), ({"minimum": 64},)])
def test_capability_rejects_invalid_range_containers(ranges):
    with pytest.raises(SchemaError, match="ShapeRange"):
        CapabilitySpec(_OP, shape_ranges=ranges)


def test_duplicate_tensor_axis_is_rejected():
    with pytest.raises(SchemaError, match="repeat a tensor axis"):
        CapabilitySpec(_OP, shape_ranges=(_RANGES[0], replace(_RANGES[0], name="other")))


def test_default_is_unrestricted_and_preserves_legacy_payload(npu_hardware):
    capability = CapabilitySpec(_OP)
    assert capability.shape_ranges == ()
    for signature in (InputSignature(_OP), _signature(((0, 1000000),))):
        assert capability.unsupported_reasons(signature, npu_hardware) == ()
    legacy_payload = {
        "op": _OP, "dtypes": [], "layouts": [], "soc_models": [],
        "min_aicore": 0, "specializations": [],
    }
    assert capability.to_dict() == legacy_payload
    legacy_digest = hashlib.sha256(json.dumps(
        legacy_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()
    assert capability.fingerprint == legacy_digest
    assert CapabilitySpec.from_dict(legacy_payload) == capability
    assert CapabilitySpec.from_dict({"op": _OP}) == capability
    descriptor = KernelDescriptor(_FALLBACK, capability=capability)
    assert "capability" not in descriptor.to_dict()
    assert descriptor_from_dict(descriptor.to_dict()).fingerprint == descriptor.fingerprint


def test_build_signature_already_carries_scale_without_new_payload_fields():
    signature = _signature()
    assert signature.shapes == ((2, 8, 96), (3, 24, 4, 96), (128, 4, 96))
    assert signature.to_dict() == {
        "op": _OP,
        "dtypes": ["float16", "float16", "float16"],
        "shapes": [[2, 8, 96], [3, 24, 4, 96], [128, 4, 96]],
        "layouts": ["", "", ""], "specializations": [],
    }


@pytest.mark.parametrize("bound, size, accepted", [
    (0, 63, False), (0, 64, True), (0, 96, True), (0, 128, True), (0, 129, False),
    (1, 0, False), (1, 1, True), (1, 4, True), (1, 8, True), (1, 9, False),
    (2, 15, False), (2, 16, True), (2, 24, True), (2, 32, True), (2, 33, False),
    (3, 0, False), (3, 1, True), (3, 128, True), (3, 256, True), (3, 257, False),
])
def test_dispatch_checks_each_scale_and_includes_both_endpoints(
    shape_registry, npu_hardware, bound, size, accepted,
):
    constraint = _RANGES[bound]
    shapes = [list(shape) for shape in _signature().shapes]
    shapes[constraint.tensor_index][constraint.axis] = size
    signature = _signature(shapes)
    decision = dispatch(signature, npu_hardware, registry=shape_registry)
    report = next(r for r in decision.reports if r.kernel_id == _LIMITED)
    assert report.accepted == accepted
    assert decision.kernel_id == (_LIMITED if accepted else _FALLBACK)
    expected = () if accepted else (
        f"{constraint.name} {size} 不在支持区间 [{constraint.minimum}, {constraint.maximum}]",
    )
    assert report.reasons == expected


@pytest.mark.parametrize("size, accepted", [(63, False), (64, True), (65, False)])
def test_singleton_range_is_exact_and_positive_axis_works(npu_hardware, size, accepted):
    capability = CapabilitySpec(_OP, shape_ranges=(ShapeRange("head_dim", 0, 2, 64, 64),))
    reasons = capability.unsupported_reasons(_signature(((2, 8, size),)), npu_hardware)
    assert (not reasons) == accepted


def test_zero_sized_axes_can_be_explicitly_supported(npu_hardware):
    capability = CapabilitySpec(_OP, shape_ranges=(ShapeRange("seq_len", 0, 0, 0, 0),))
    assert capability.unsupported_reasons(_signature(((0, 8, 64),)), npu_hardware) == ()


@pytest.mark.parametrize("shapes, bound, detail", [
    ((), _RANGES[0], "缺少输入张量 0 的 shape"),
    (((2, 8, 64),), _RANGES[2], "缺少输入张量 1 的 shape"),
    (((),), _RANGES[0], "无轴 -1"),
    (((2, 8, 64),), ShapeRange("head_dim", 0, 3, 64, 128), "无轴 3"),
    (((2, 8, 64),), ShapeRange("head_dim", 0, -4, 64, 128), "无轴 -4"),
])
def test_missing_shapes_or_axes_reject_instead_of_silently_passing(
    npu_hardware, shapes, bound, detail,
):
    reasons = CapabilitySpec(_OP, shape_ranges=(bound,)).unsupported_reasons(
        _signature(shapes), npu_hardware,
    )
    assert len(reasons) == 1
    assert bound.name in reasons[0] and detail in reasons[0]
    assert "无法检查支持区间" in reasons[0]


def test_explain_contains_readable_shape_rejection(shape_registry, npu_hardware):
    signature = _signature(((2, 8, 256), (3, 24, 4, 96), (128, 4, 96)))
    explanation = explain_dispatch(signature, npu_hardware, registry=shape_registry)
    assert f"[reject] {_LIMITED}" in explanation
    assert "head_dim 256 不在支持区间 [64, 128]" in explanation
    assert f"[accept] {_FALLBACK}" in explanation


def test_explain_keeps_all_capability_reasons(shape_registry, npu_hardware):
    signature = replace(
        _signature(((2, 8, 256), (3, 64, 4, 96), (512, 4, 96))), dtypes=("int8",),
    )
    decision = dispatch(signature, npu_hardware, registry=shape_registry)
    report = next(r for r in decision.reports if r.kernel_id == _LIMITED)
    explanation = explain_dispatch(signature, npu_hardware, registry=shape_registry)
    assert len(report.reasons) == 4
    for reason in (
        "dtype int8 不在支持集 ['float16']",
        "head_dim 256 不在支持区间 [64, 128]",
        "page_size 64 不在支持区间 [16, 32]",
        "seq_len 512 不在支持区间 [1, 256]",
    ):
        assert reason in report.reasons
        assert reason in explanation


def test_explicit_kernel_rejects_out_of_range_and_accepts_endpoint(shape_registry, npu_hardware):
    signature = _signature(((2, 8, 128), (3, 24, 4, 96), (128, 4, 96)))
    assert dispatch(signature, npu_hardware, kernel_id=_LIMITED, registry=shape_registry).kernel_id == _LIMITED
    with pytest.raises(DispatchError, match="head_dim 256 不在支持区间 \\[64, 128\\]"):
        dispatch(
            replace(signature, shapes=((2, 8, 256), *signature.shapes[1:])),
            npu_hardware, kernel_id=_LIMITED, registry=shape_registry,
        )


def test_no_survivors_explains_shape_failure(shape_registry, npu_hardware):
    with pytest.raises(DispatchError, match="head_dim 256 不在支持区间 \\[64, 128\\]"):
        explain_dispatch(
            _signature(((2, 8, 256), (3, 24, 4, 96), (128, 4, 96))), npu_hardware,
            backend="ascendc", registry=shape_registry,
        )


def test_actual_tensor_shape_cannot_be_overridden_by_specializations(npu_hardware):
    capability = CapabilitySpec(_OP, shape_ranges=(_RANGES[0],))
    signature = build_signature(
        _OP, (SimpleNamespace(dtype="float16", shape=(2, 8, 256)),),
        specializations={"head_dim": 64},
    )
    assert capability.unsupported_reasons(signature, npu_hardware) == (
        "head_dim 256 不在支持区间 [64, 128]",
    )


def test_shape_ranges_and_enumerated_specializations_both_apply(npu_hardware):
    capability = CapabilitySpec(
        _OP, specializations=(("head_dim", ("64", "128")),), shape_ranges=(_RANGES[0],),
    )
    signature = replace(_signature(), specializations=(("head_dim", "96"),))
    assert capability.unsupported_reasons(signature, npu_hardware) == (
        "特化参数 head_dim=96 不在允许集 ['128', '64']",
    )


def test_shape_changes_do_not_reuse_incompatible_cached_decisions(shape_registry, npu_hardware):
    inside = _signature()
    outside = replace(inside, shapes=((2, 8, 256), *inside.shapes[1:]))
    first = dispatch(inside, npu_hardware, registry=shape_registry)
    second = dispatch(outside, npu_hardware, registry=shape_registry)
    third = dispatch(inside, npu_hardware, registry=shape_registry)
    assert first.kernel_id == third.kernel_id == _LIMITED
    assert second.kernel_id == _FALLBACK
    assert not first.cache_hit and not second.cache_hit and third.cache_hit


def test_ranges_survive_json_manifest_round_trip_and_change_fingerprints(shape_registry, tmp_path):
    descriptor = shape_registry.get(_LIMITED)
    payload = json.loads(json.dumps(descriptor.to_dict()))
    assert payload["capability"]["shape_ranges"][0] == {
        "name": "head_dim", "tensor_index": 0, "axis": -1, "minimum": 64, "maximum": 128,
    }
    restored = descriptor_from_dict(payload)
    assert restored == descriptor
    assert restored.fingerprint == descriptor.fingerprint
    assert restored.capability.fingerprint == descriptor.capability.fingerprint
    manifest_path = tmp_path / "kernels.json"
    manifest_path.write_text(json.dumps({
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "generated_at": "2026-10-22", "kernels": [payload],
    }))
    assert load_kernel_manifest(manifest_path) == (descriptor,)
    for ranges in ((), (replace(_RANGES[0], maximum=256), *_RANGES[1:])):
        changed = replace(descriptor, capability=replace(descriptor.capability, shape_ranges=ranges))
        assert changed.fingerprint != descriptor.fingerprint
        assert changed.capability.fingerprint != descriptor.capability.fingerprint
    reordered = replace(descriptor, capability=replace(descriptor.capability, shape_ranges=reversed(_RANGES)))
    assert reordered.fingerprint == descriptor.fingerprint


@pytest.mark.parametrize("capability", [
    None,
    "invalid",
    {"op": _OP, "unknown": True},
    {"op": _OP, "shape_ranges": None},
    {"op": _OP, "shape_ranges": [{"minimum": 64}]},
    {"op": "different", "shape_ranges": []},
])
def test_manifest_rejects_invalid_capability(capability):
    with pytest.raises(SchemaError):
        descriptor_from_dict({
            "kernel_id": _LIMITED, "op": _OP, "backend": "ascendc", "capability": capability,
        })


def test_range_api_is_tracked_and_survives_ledger_refresh():
    from flashinfer.parity import ParityEntry, load_packaged_manifest
    from tools.update_parity_inventory import annotate

    class UpstreamWithoutRuntimeContracts:
        def resolve(self, symbol):
            return None

    entries = {entry.local: entry for entry in load_packaged_manifest("all").entries}
    for name in ("CapabilitySpec", "ShapeRange"):
        entry = entries["flashinfer.runtime.schema." + name]
        assert entry.api_kind == "npu_extension"
        assert entry.implementation_status == "framework"
        refreshed = ParityEntry.from_dict(annotate(entry.to_dict(), UpstreamWithoutRuntimeContracts()))
        assert refreshed == entry
