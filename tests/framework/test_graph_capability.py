"""T6.5：入图静态声明、按 ID 查询及 eager 选型兼容性。"""

from dataclasses import FrozenInstanceError, replace
import json
from unittest.mock import Mock

import pytest

from flashinfer.runtime import GraphCapability
from flashinfer.runtime.dispatch import dispatch
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.manifest import (
    KERNEL_MANIFEST_SCHEMA_VERSION, descriptor_from_dict, load_kernel_manifest,
)
from flashinfer.runtime.registry import KernelRegistry, _builtin_descriptors
from flashinfer.runtime.schema import (
    Backend, DeviceCapability, DispatchError, KernelDescriptor, SchemaError, WorkloadSpec,
)
from flashinfer.runtime.signature import InputSignature


_OP = "attention.decode"
_ID = "cann/test/graph"
_PARAMS = ("actual_seq_lengths_kv", "tiling")
_DECLARATIONS = (
    (GraphCapability.CAPTURE_SAFE, ()),
    (GraphCapability.NEEDS_PARAM_UPDATE, _PARAMS),
    (GraphCapability.NOT_CAPTURABLE, ()),
)


def _descriptor(**kwargs):
    return KernelDescriptor(_ID, op=_OP, backend=Backend.CANN, **kwargs)


@pytest.mark.parametrize("capability, params", _DECLARATIONS)
@pytest.mark.parametrize("as_string", (False, True))
def test_all_graph_declarations_accept_enums_and_strings(capability, params, as_string):
    descriptor = _descriptor(
        graph_capability=capability.value if as_string else capability,
        graph_param_updates=params,
    )
    assert descriptor.graph_capability is capability
    assert descriptor.graph_param_updates == params


def test_default_is_conservative_and_preserves_legacy_serialization():
    descriptor = _descriptor()
    assert descriptor.graph_capability is GraphCapability.NOT_CAPTURABLE
    assert descriptor.graph_param_updates == ()
    legacy_payload = descriptor.to_dict()
    assert "graph_capability" not in legacy_payload
    assert "graph_param_updates" not in legacy_payload
    restored = descriptor_from_dict(json.loads(json.dumps(legacy_payload)))
    assert restored == descriptor
    assert restored.fingerprint == descriptor.fingerprint
    explicit_default = _descriptor(graph_capability="not_capturable", graph_param_updates=[])
    assert explicit_default.to_dict() == legacy_payload
    assert explicit_default.fingerprint == descriptor.fingerprint


def test_update_parameter_names_are_copied_frozen_and_hashable():
    params = list(_PARAMS)
    descriptor = _descriptor(graph_capability="needs_param_update", graph_param_updates=params)
    params.append("unrelated")
    assert descriptor.graph_param_updates == _PARAMS
    assert hash(descriptor) == hash(_descriptor(
        graph_capability="needs_param_update", graph_param_updates=_PARAMS,
    ))
    with pytest.raises(FrozenInstanceError):
        descriptor.graph_param_updates = ()
    with pytest.raises(FrozenInstanceError):
        descriptor.graph_capability = GraphCapability.CAPTURE_SAFE


@pytest.mark.parametrize("value", ["", "auto", "CAPTURE_SAFE", "safe", None, True, 1, (), {}, []])
def test_invalid_graph_capabilities_raise_schema_error(value):
    with pytest.raises(SchemaError, match="graph_capability"):
        _descriptor(graph_capability=value)
    with pytest.raises(SchemaError, match="graph_capability"):
        descriptor_from_dict({**_descriptor().to_dict(), "graph_capability": value})


@pytest.mark.parametrize("value", [
    None, "seq_len", 1, {"seq_len": 0}, {"seq_len"},
    (None,), (1,), ("",), (" ",), ("seq len",), ("seq_len\0",),
    ("seq_len", "seq_len"), (["seq_len"],),
])
def test_invalid_update_parameter_declarations_raise_schema_error(value):
    with pytest.raises(SchemaError, match="graph_param_updates"):
        _descriptor(graph_capability="needs_param_update", graph_param_updates=value)
    with pytest.raises(SchemaError, match="graph_param_updates"):
        descriptor_from_dict({
            **_descriptor().to_dict(), "graph_capability": "needs_param_update",
            "graph_param_updates": value,
        })


@pytest.mark.parametrize("params", ((), []))
def test_needs_param_update_requires_parameter_names(params):
    with pytest.raises(SchemaError, match="needs_param_update requires non-empty graph_param_updates"):
        _descriptor(graph_capability="needs_param_update", graph_param_updates=params)
    with pytest.raises(SchemaError, match="needs_param_update requires non-empty graph_param_updates"):
        descriptor_from_dict({**_descriptor().to_dict(), "graph_capability": "needs_param_update"})


@pytest.mark.parametrize("capability", ("capture_safe", "not_capturable"))
def test_other_graph_capabilities_reject_update_parameters(capability):
    with pytest.raises(SchemaError, match="requires graph_capability=needs_param_update"):
        _descriptor(graph_capability=capability, graph_param_updates=_PARAMS)
    with pytest.raises(SchemaError, match="requires graph_capability=needs_param_update"):
        descriptor_from_dict({
            **_descriptor().to_dict(), "graph_capability": capability,
            "graph_param_updates": list(_PARAMS),
        })


@pytest.mark.parametrize("capability, params", _DECLARATIONS)
def test_query_returns_graph_capability_and_parameters_without_execution(capability, params):
    runner = Mock(side_effect=AssertionError("metadata query must not execute the kernel"))
    descriptor = _descriptor(graph_capability=capability, graph_param_updates=params, runner=runner)
    registry = KernelRegistry((descriptor,))
    registry.freeze()
    revision = registry.revision
    assert registry.graph_requirements(_ID) == (capability, params)
    assert registry.revision == revision
    assert registry.descriptors() == (descriptor,)
    runner.assert_not_called()
    with pytest.raises(KeyError, match="未注册的 kernel_id：missing"):
        registry.graph_requirements("missing")


def test_only_proven_record_declares_capture_and_keeps_others_conservative():
    from tools.audit_graph_replay import audit_declarations
    from tests.production_inventory import LEGACY_KERNEL_IDS

    registry = KernelRegistry(_builtin_descriptors())
    audit_declarations(registry.descriptors())
    assert sorted(d.kernel_id for d in registry.descriptors()) == sorted(LEGACY_KERNEL_IDS)
    for descriptor in registry.descriptors():
        if descriptor.kernel_id == "cann/attention/single_decode_fia_v1":
            assert registry.graph_requirements(descriptor.kernel_id) == (
                GraphCapability.NEEDS_PARAM_UPDATE, ("actual_seq_lengths_kv",),
            )
            assert descriptor.graph_replay.validated_kernel_id == descriptor.kernel_id
            assert descriptor.graph_replay.validation_tests
            restored = descriptor_from_dict(json.loads(json.dumps(descriptor.to_dict())))
            assert restored.graph_replay == descriptor.graph_replay
            assert restored.fingerprint == descriptor.fingerprint
            continue
        assert registry.graph_requirements(descriptor.kernel_id) == (
            GraphCapability.NOT_CAPTURABLE, (),
        )
        assert "graph_capability" not in descriptor.to_dict()
        assert "graph_param_updates" not in descriptor.to_dict()


@pytest.mark.parametrize("capability, params", _DECLARATIONS)
def test_graph_declarations_survive_json_and_manifest_round_trips(capability, params, tmp_path):
    descriptor = _descriptor(graph_capability=capability, graph_param_updates=params)
    payload = json.loads(json.dumps(descriptor.to_dict()))
    if capability == GraphCapability.NOT_CAPTURABLE:
        assert "graph_capability" not in payload
    else:
        assert payload["graph_capability"] == capability.value
    if params:
        assert payload["graph_param_updates"] == list(params)
    else:
        assert "graph_param_updates" not in payload
    restored = descriptor_from_dict(payload)
    assert restored == descriptor
    assert restored.fingerprint == descriptor.fingerprint
    manifest_path = tmp_path / "kernels.json"
    manifest_path.write_text(json.dumps({
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "generated_at": "2026-10-22", "kernels": [payload],
    }))
    assert load_kernel_manifest(manifest_path) == (descriptor,)
    assert KernelRegistry((restored,)).graph_requirements(_ID) == (capability, params)


def test_non_default_graph_declarations_participate_in_fingerprints():
    default = _descriptor()
    safe = _descriptor(graph_capability="capture_safe")
    updatable = _descriptor(graph_capability="needs_param_update", graph_param_updates=_PARAMS)
    changed_params = replace(updatable, graph_param_updates=("seq_len",))
    assert len({d.fingerprint for d in (default, safe, updatable, changed_params)}) == 4


def _auto_result(registry, signature, hardware, allow_reference):
    try:
        return dispatch(
            signature, hardware, backend="auto", allow_reference=allow_reference,
            registry=registry,
        )
    except DispatchError as error:
        return str(error)


@pytest.mark.parametrize("capability, params", _DECLARATIONS)
def test_graph_declarations_do_not_change_auto_for_any_existing_record(
    capability, params, monkeypatch,
):
    # 不让不同 registry 的同 revision 缓存使前后对照得到假阳性。
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1")
    original = KernelRegistry(_builtin_descriptors())
    annotated = KernelRegistry(
        replace(d, graph_capability=capability, graph_param_updates=params, graph_replay=None)
        for d in original.descriptors()
    )
    for descriptor in original.descriptors():
        spec = descriptor.capability
        signature = InputSignature(
            descriptor.op,
            dtypes=(spec.dtypes[0] if spec.dtypes else "float16",),
            shapes=((2, 8, 128),), layouts=(spec.layouts[0] if spec.layouts else "",),
            specializations=tuple((key, allowed[0]) for key, allowed in spec.specializations),
        )
        for soc in ("Ascend950", "Ascend910B"):
            hardware = HardwareSignature(soc_model=soc, aicore_count=32)
            for allow_reference in (False, True):
                assert _auto_result(original, signature, hardware, allow_reference) == _auto_result(
                    annotated, signature, hardware, allow_reference,
                ), (descriptor.kernel_id, soc, allow_reference)


@pytest.mark.parametrize("capability, params", _DECLARATIONS)
def test_graph_declarations_do_not_change_manifest_selection(capability, params):
    preferred = _descriptor(priority=10)
    fallback = _descriptor(graph_capability="capture_safe")
    fallback = replace(fallback, kernel_id="cann/test/fallback")
    original = KernelRegistry((preferred, fallback))
    annotated = KernelRegistry((
        replace(preferred, graph_capability=capability, graph_param_updates=params), fallback,
    ))
    workload = WorkloadSpec(_OP)
    device = DeviceCapability("Ascend910B")
    assert original.select(workload, device).kernel_id == _ID
    assert annotated.select(workload, device).kernel_id == _ID
    assert original.explain(workload, device) == annotated.explain(workload, device)


def test_graph_api_is_tracked_and_survives_ledger_refresh():
    from flashinfer.parity import ParityEntry, load_packaged_manifest
    from tools.update_parity_inventory import annotate

    class UpstreamWithoutRuntimeContracts:
        def resolve(self, symbol):
            return None

    entries = {entry.local: entry for entry in load_packaged_manifest("all").entries}
    for symbol in (
        "flashinfer.runtime.schema.GraphCapability",
        "flashinfer.runtime.schema.KernelDescriptor",
        "flashinfer.runtime.registry.KernelRegistry.graph_requirements",
    ):
        entry = entries[symbol]
        assert entry.api_kind == "npu_extension"
        assert entry.implementation_status == "framework"
        assert entry.semantic_status == "intentionally_different"
        refreshed = ParityEntry.from_dict(annotate(entry.to_dict(), UpstreamWithoutRuntimeContracts()))
        assert refreshed == entry
