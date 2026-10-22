"""Manifest selection must enforce the policies shared with tensor dispatch."""

from dataclasses import replace
import importlib
from itertools import permutations
from unittest.mock import Mock

import pytest

from flashinfer.runtime.dispatch import dispatch
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.plan import build_plan
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import (
    Backend, CapabilitySpec, CompileMode, DeviceCapability, DispatchError,
    KernelConstraints, KernelDescriptor, ShapeRange, WorkloadSpec, WorkspaceFormula,
)
from flashinfer.runtime.signature import InputSignature


_OP = "test.selection"
_WORKLOAD = WorkloadSpec(
    _OP, dtypes=("float16",), layouts=("NHD",), attributes=(("head_dim", "64"),),
)
_DEVICE = DeviceCapability("Ascend910B", ai_core_count=20)
_HARDWARE = HardwareSignature("Ascend910B", aicore_count=20)
_SIGNATURE = InputSignature(
    _OP, dtypes=_WORKLOAD.dtypes, layouts=_WORKLOAD.layouts,
    specializations=(*_WORKLOAD.attributes, ("causal", "False"),
                     ("pos_encoding", "NONE"), ("deterministic", "False")),
)


@pytest.fixture(autouse=True)
def _selection_environment(monkeypatch):
    for name in ("BACKEND", "ALLOW_CUSTOM_KERNELS", "JIT_DISABLE", "DISPATCH_ORDER"):
        monkeypatch.delenv("FLASHINFER_NPU_" + name, raising=False)
    # Comparisons use independent registries; do not reuse another registry's decision.
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1")


def _descriptor(backend=Backend.CANN, name="preferred", **kwargs):
    namespace = "vendor" if backend == Backend.EXTERNAL else backend.value
    return KernelDescriptor(f"{namespace}/selection/{name}", op=_OP,
                            backend=backend, **kwargs)


def _registry(*descriptors):
    registry = KernelRegistry()
    for descriptor in descriptors:
        if descriptor.backend == Backend.CUSTOM:
            registry.register_custom(descriptor)
        elif descriptor.backend == Backend.EXTERNAL:
            registry.register_external(descriptor)
        else:
            registry.register(descriptor)
    return registry


@pytest.mark.parametrize("backend", [None, "auto"])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("via_plan", [False, True])
def test_custom_auto_requires_opt_in(backend, tuned, via_plan):
    custom = _descriptor(Backend.CUSTOM, compile_mode=CompileMode.AOT, priority=999)
    registry = _registry(custom)
    with pytest.raises(DispatchError, match="FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS"):
        dispatch(_SIGNATURE, _HARDWARE, backend=backend, registry=registry)
    selection = dict(backend=backend,
                     tuned_kernel_ids=(custom.kernel_id,) if tuned else ())
    with pytest.raises(DispatchError, match="FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS"):
        if via_plan:
            build_plan(registry, _WORKLOAD, _DEVICE, **selection)
        else:
            registry.select(_WORKLOAD, _DEVICE, **selection)


def test_custom_opt_in_and_explicit_backend_remain_supported(monkeypatch):
    custom = _descriptor(Backend.CUSTOM, compile_mode=CompileMode.AOT)
    registry = _registry(custom)
    assert registry.select(_WORKLOAD, _DEVICE, backend="custom") is custom
    monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "1")
    assert registry.select(_WORKLOAD, _DEVICE) is custom
    monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "0")
    with pytest.raises(DispatchError, match="FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS"):
        registry.select(_WORKLOAD, _DEVICE)
    assert registry.select(_WORKLOAD, _DEVICE, backend="custom") is custom


@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("via_plan", [False, True])
def test_jit_disable_applies_to_manifest_selection(monkeypatch, backend, via_plan):
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    jit = _descriptor(Backend.TRITON, compile_mode=CompileMode.JIT)
    registry = _registry(jit)
    with pytest.raises(DispatchError, match="FLASHINFER_NPU_JIT_DISABLE"):
        dispatch(_SIGNATURE, _HARDWARE, backend=backend, registry=registry)
    selection = dict(backend=backend, tuned_kernel_ids=(jit.kernel_id,))
    with pytest.raises(DispatchError, match="FLASHINFER_NPU_JIT_DISABLE"):
        if via_plan:
            build_plan(registry, _WORKLOAD, _DEVICE, **selection)
        else:
            registry.select(_WORKLOAD, _DEVICE, **selection)


@pytest.mark.parametrize("mode", list(CompileMode))
def test_jit_disable_uses_declared_compile_mode(monkeypatch, mode):
    descriptor = _descriptor(Backend.TRITON, compile_mode=mode)
    registry = _registry(descriptor)
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    if mode == CompileMode.JIT:
        with pytest.raises(DispatchError):
            registry.select(_WORKLOAD, _DEVICE)
        monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    assert registry.select(_WORKLOAD, _DEVICE) is descriptor


def test_backend_none_honors_global_default(monkeypatch):
    cann = _descriptor(priority=100)
    triton = _descriptor(Backend.TRITON, priority=0)
    registry = _registry(cann, triton)
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "triton")
    assert dispatch(_SIGNATURE, _HARDWARE, backend=None,
                    registry=registry).kernel_id == triton.kernel_id
    assert registry.select(_WORKLOAD, _DEVICE) is triton
    assert build_plan(registry, _WORKLOAD, _DEVICE).kernel is triton
    assert registry.select(_WORKLOAD, _DEVICE, backend="auto") is cann
    assert registry.select(_WORKLOAD, _DEVICE, backend="cann") is cann


@pytest.mark.parametrize("backend", ["reference", Backend.REFERENCE, None])
def test_explicit_reference_backend_is_opt_in(monkeypatch, backend):
    reference = _descriptor(Backend.REFERENCE)
    registry = _registry(reference)
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "reference")
    assert dispatch(_SIGNATURE, _HARDWARE, backend=backend,
                    registry=registry).kernel_id == reference.kernel_id
    assert registry.select(_WORKLOAD, _DEVICE, backend=backend) is reference
    with pytest.raises(DispatchError, match="opt-in"):
        registry.select(_WORKLOAD, _DEVICE, backend="auto")
    assert registry.select(_WORKLOAD, _DEVICE, backend="auto", allow_reference=True) is reference


@pytest.mark.parametrize("limits, reason", [
    ({"dtypes": ("bfloat16",)}, "dtype"),
    ({"layouts": ("HND",)}, "layout"),
    ({"soc_models": ("Ascend950",)}, "SoC"),
    ({"min_aicore": 21}, "AI Core"),
    ({"specializations": (("head_dim", ("128",)),)}, "head_dim"),
    ({"specializations": (("causal", ("True",)),)}, "causal"),
    ({"specializations": (("pos_encoding", ("ROPE_LLAMA",)),)}, "pos_encoding"),
    ({"specializations": (("deterministic", ("True",)),)}, "deterministic"),
    ({"op": "another.op"}, "op"),
])
def test_select_checks_shared_capability(limits, reason):
    descriptor = _descriptor(capability=CapabilitySpec(**{"op": _OP, **limits}))
    registry = _registry(descriptor)
    with pytest.raises(DispatchError, match=reason):
        dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    with pytest.raises(DispatchError, match=reason):
        registry.select(_WORKLOAD, _DEVICE, tuned_kernel_ids=(descriptor.kernel_id,))
    assert any(reason in item for item in registry.explain(_WORKLOAD, _DEVICE)[0].reasons)


def test_matching_shared_capability_is_accepted():
    descriptor = _descriptor(capability=CapabilitySpec(
        _OP, dtypes=("float16",), layouts=("NHD",), soc_models=("Ascend910B",),
        min_aicore=20, specializations=(("head_dim", ("64",)), ("causal", ("False",))),
    ))
    registry = _registry(descriptor)
    assert registry.select(_WORKLOAD, _DEVICE) is descriptor
    assert registry.explain(_WORKLOAD, _DEVICE)[0].reasons == ()


def test_workload_scalar_specializations_cannot_be_overridden_by_attributes():
    workload = replace(_WORKLOAD, causal=True,
                       attributes=(*_WORKLOAD.attributes, ("causal", "False")))
    descriptor = _descriptor(capability=CapabilitySpec(
        _OP, specializations=(("causal", ("False",)),),
    ))
    with pytest.raises(DispatchError, match="causal=True"):
        _registry(descriptor).select(workload, _DEVICE)


def test_duplicate_attributes_do_not_hide_unsupported_specializations():
    workload = replace(_WORKLOAD, attributes=(("head_dim", "128"), ("head_dim", "64")))
    descriptor = _descriptor(capability=CapabilitySpec(
        _OP, specializations=(("head_dim", ("64",)),),
    ))
    with pytest.raises(DispatchError, match="head_dim=128"):
        _registry(descriptor).select(workload, _DEVICE)


def test_default_tiers_outrank_priority():
    cann = _descriptor(priority=-100)
    aot = _descriptor(Backend.ASCENDC, compile_mode=CompileMode.AOT, priority=999)
    registry = _registry(aot, cann)
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).kernel_id == cann.kernel_id
    assert registry.select(_WORKLOAD, _DEVICE) is cann
    assert build_plan(registry, _WORKLOAD, _DEVICE).kernel is cann


def test_configured_tiers_control_manifest_selection(monkeypatch):
    descriptors = (
        _descriptor(Backend.EXTERNAL), _descriptor(),
        _descriptor(Backend.ASCENDC, compile_mode=CompileMode.AOT),
        _descriptor(Backend.TRITON, compile_mode=CompileMode.JIT),
        _descriptor(Backend.REFERENCE),
    )
    groups = ("external", "cann", "aot", "jit", "reference")
    by_group = dict(zip(groups, descriptors))
    registry = _registry(*descriptors)
    for order in permutations(groups):
        monkeypatch.setenv("FLASHINFER_NPU_DISPATCH_ORDER", ",".join(order))
        expected = by_group[order[0]]
        assert dispatch(_SIGNATURE, _HARDWARE, registry=registry,
                        allow_reference=True).kernel_id == expected.kernel_id
        assert registry.select(_WORKLOAD, _DEVICE, allow_reference=True) is expected
    monkeypatch.setenv("FLASHINFER_NPU_DISPATCH_ORDER", "reference,jit,aot,cann,external")
    assert registry.select(_WORKLOAD, _DEVICE) is by_group["jit"]


def test_invalid_order_is_rejected_by_select(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_DISPATCH_ORDER", "cann")
    with pytest.raises(ValueError, match="FLASHINFER_NPU_DISPATCH_ORDER"):
        _registry(_descriptor()).select(_WORKLOAD, _DEVICE)


@pytest.mark.parametrize("restriction", [
    "custom", "jit", "capability", "target", "manifest", "reference", "backend", "unknown",
])
def test_tuning_cannot_revive_filtered_candidates(monkeypatch, restriction):
    fallback = _descriptor(name="fallback")
    preferred = _descriptor(priority=100)
    backend = "auto"
    if restriction == "custom":
        preferred = _descriptor(Backend.CUSTOM, priority=100)
    elif restriction == "jit":
        preferred = replace(preferred, compile_mode=CompileMode.JIT)
        monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    elif restriction == "capability":
        preferred = replace(preferred, capability=CapabilitySpec(_OP, min_aicore=21))
    elif restriction == "target":
        preferred = replace(preferred, target_soc="Ascend950")
    elif restriction == "manifest":
        preferred = replace(preferred, constraints=KernelConstraints(required_features=("missing",)))
    elif restriction == "reference":
        preferred = _descriptor(Backend.REFERENCE, priority=100)
    elif restriction == "backend":
        preferred = _descriptor(Backend.TRITON, priority=100)
        backend = "cann"
    tuned_id = "missing/kernel" if restriction == "unknown" else preferred.kernel_id
    descriptors = (fallback,) if restriction == "unknown" else (fallback, preferred)
    registry = _registry(*descriptors)
    assert registry.select(_WORKLOAD, _DEVICE, backend=backend,
                           tuned_kernel_ids=(tuned_id, fallback.kernel_id)) is fallback


def test_manifest_tuning_and_tie_break_are_preserved():
    descriptors = tuple(_descriptor(name=name, priority=priority) for name, priority in (
        ("z", 10), ("a", 10), ("low", 0),
    ))
    for registration in permutations(descriptors):
        registry = _registry(*registration)
        assert registry.select(_WORKLOAD, _DEVICE) is descriptors[1]
        assert registry.select(_WORKLOAD, _DEVICE,
                               tuned_kernel_ids=("missing", descriptors[2].kernel_id)) is descriptors[2]


def test_tensor_only_limits_do_not_guess_workload_shapes_or_layouts(monkeypatch):
    descriptor = _descriptor(capability=CapabilitySpec(
        _OP, dtypes=("float16",), physical_layouts=("NZ",),
        shape_ranges=(ShapeRange("head_dim", 0, -1, 64, 128),),
    ))
    registry = _registry(descriptor)
    workload = replace(_WORKLOAD, static_dims=(999,), dynamic_bounds=(999,))
    query = Mock(side_effect=AssertionError("incomplete manifest data is not a tuning key"))
    monkeypatch.setattr(importlib.import_module("flashinfer.runtime.dispatch"),
                        "query_tuning_record", query)
    assert registry.select(workload, _DEVICE) is descriptor
    assert registry.explain(workload, _DEVICE)[0].reasons == ()
    query.assert_not_called()
    # Skipping tensor-only limits must still enforce the shared dtype restriction.
    with pytest.raises(DispatchError, match="dtype"):
        registry.select(replace(workload, dtypes=("int8",)), _DEVICE)


def test_explain_remains_policy_free_and_plan_keeps_workspace(monkeypatch):
    descriptor = _descriptor(
        Backend.CUSTOM, compile_mode=CompileMode.JIT,
        workspace=WorkspaceFormula(constant_bytes=33),
        int_workspace=WorkspaceFormula(constant_bytes=1),
    )
    registry = _registry(descriptor)
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    assert registry.explain(_WORKLOAD, _DEVICE)[0].reasons == ()
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    plan = build_plan(registry, _WORKLOAD, _DEVICE, backend="custom")
    assert plan.kernel is descriptor
    assert (plan.workspace_bytes, plan.int_workspace_bytes) == (64, 32)
    plan.validate_runtime(_WORKLOAD, _DEVICE)
