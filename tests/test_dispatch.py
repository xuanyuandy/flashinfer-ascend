"""dispatcher 行为（§5.3 / §5.4.3 / §10.5 / §12，全部 host 可测）。"""

from dataclasses import replace
import json

import pytest

from flashinfer.runtime.dispatch import dispatch, explain_dispatch
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import (
    Backend,
    CapabilitySpec,
    CompileMode,
    DispatchError,
    KernelDescriptor,
)
from flashinfer.runtime.signature import InputSignature

_SIG_FP16 = InputSignature(op="attention.decode", dtypes=("float16",))
_SIG_INT8 = InputSignature(op="attention.decode", dtypes=("int8",))


def test_auto_prefers_self_built_aot(decode_registry, npu_hardware):
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert decision.kernel_id == "ascendc/attention/decode_v1"
    assert decision.backend == Backend.ASCENDC
    assert decision.fallback_chain == (
        "cann/attention/decode_aclnn_v1",
        "triton/attention/decode_v1",
    )


def test_capability_failure_falls_through_to_cann(decode_registry, npu_hardware):
    # ascendc 只支持 fp16/bf16，int8 被淘汰 → CANN 胜出
    decision = dispatch(_SIG_INT8, npu_hardware, registry=decode_registry)
    assert decision.kernel_id == "cann/attention/decode_aclnn_v1"
    report = {r.kernel_id: r for r in decision.reports}[
        "ascendc/attention/decode_v1"
    ]
    assert not report.accepted
    assert any("int8" in reason for reason in report.reasons)


def test_explicit_backend_only_narrows_candidates(decode_registry, npu_hardware):
    decision = dispatch(
        _SIG_FP16, npu_hardware, backend="triton", registry=decode_registry
    )
    assert decision.kernel_id == "triton/attention/decode_v1"


def test_reference_excluded_from_auto_by_default(decode_registry, npu_hardware):
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    report = {r.kernel_id: r for r in decision.reports}[
        "reference/attention/decode_v1"
    ]
    assert not report.accepted
    assert any("opt-in" in reason for reason in report.reasons)


def test_reference_available_via_explicit_opt_in(decode_registry, npu_hardware):
    decision = dispatch(
        _SIG_FP16, npu_hardware, backend="reference", registry=decode_registry
    )
    assert decision.kernel_id == "reference/attention/decode_v1"


def test_explicit_kernel_id_locks(decode_registry, npu_hardware):
    decision = dispatch(
        _SIG_FP16,
        npu_hardware,
        kernel_id="triton/attention/decode_v1",
        registry=decode_registry,
    )
    assert decision.kernel_id == "triton/attention/decode_v1"
    assert decision.backend == Backend.TRITON


def test_explicit_kernel_id_capability_mismatch_raises(
    decode_registry, npu_hardware
):
    with pytest.raises(DispatchError, match="int8"):
        dispatch(
            _SIG_INT8,
            npu_hardware,
            kernel_id="ascendc/attention/decode_v1",
            registry=decode_registry,
        )


def test_no_candidate_raises_with_reasons(decode_registry, npu_hardware):
    unknown = InputSignature(op="attention.prefill", dtypes=("float16",))
    with pytest.raises(DispatchError) as excinfo:
        dispatch(unknown, npu_hardware, registry=decode_registry)
    assert "无可用 kernel" in str(excinfo.value)


def test_jit_disabled_filters_jit_kernel(
    decode_registry, npu_hardware, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    report = {r.kernel_id: r for r in decision.reports}[
        "triton/attention/decode_v1"
    ]
    assert not report.accepted


def _custom_descriptor(priority: int = 0) -> KernelDescriptor:
    return KernelDescriptor(
        kernel_id="custom/attention/decode_mine",
        backend=Backend.CUSTOM,
        capability=CapabilitySpec(op="attention.decode"),
        compile_mode=CompileMode.AOT,
        priority=priority,
    )


def test_custom_kernel_excluded_from_auto_by_default(
    decode_registry, npu_hardware
):
    decode_registry.register_custom(_custom_descriptor())
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert decision.kernel_id == "ascendc/attention/decode_v1"


def test_custom_kernel_competes_when_enabled(
    decode_registry, npu_hardware, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "1")
    decode_registry.register_custom(_custom_descriptor(priority=99))
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    # custom 为 AOT（tier 0）且 priority 更高 → 同层胜出
    assert decision.kernel_id == "custom/attention/decode_mine"


def test_custom_kernel_always_available_via_explicit_kernel_id(
    decode_registry, npu_hardware
):
    decode_registry.register_custom(_custom_descriptor())
    decision = dispatch(
        _SIG_FP16,
        npu_hardware,
        kernel_id="custom/attention/decode_mine",
        registry=decode_registry,
    )
    assert decision.kernel_id == "custom/attention/decode_mine"


def test_custom_registration_rules(decode_registry):
    with pytest.raises(ValueError, match="custom/"):
        decode_registry.register_custom(
            KernelDescriptor(
                kernel_id="attention/decode_mine",  # 缺 custom/ 前缀
                backend=Backend.CUSTOM,
                capability=CapabilitySpec(op="attention.decode"),
            )
        )
    with pytest.raises(ValueError, match="backend 必须为 custom"):
        decode_registry.register_custom(
            KernelDescriptor(
                kernel_id="custom/attention/decode_mine",
                backend=Backend.ASCENDC,  # 自定义不得冒用内置 backend
                capability=CapabilitySpec(op="attention.decode"),
            )
        )
    decode_registry.register_custom(_custom_descriptor())
    with pytest.raises(ValueError, match="重复"):
        decode_registry.register_custom(_custom_descriptor())


def test_builtin_registration_blocked_after_freeze(decode_registry):
    decode_registry.freeze()
    with pytest.raises(RuntimeError, match="冻结"):
        decode_registry.register(
            KernelDescriptor(
                kernel_id="ascendc/attention/decode_v2",
                backend=Backend.ASCENDC,
                capability=CapabilitySpec(op="attention.decode"),
            )
        )


def test_decision_cache_hit_and_registry_invalidation(
    decode_registry, npu_hardware
):
    first = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert not first.cache_hit
    second = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert second.cache_hit
    assert second.kernel_id == first.kernel_id
    # 注册新 kernel → revision 变化 → 旧决策缓存自动失效
    decode_registry.register_custom(_custom_descriptor())
    third = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert not third.cache_hit


def test_decision_cache_can_be_disabled(
    decode_registry, npu_hardware, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1")
    dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    again = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert not again.cache_hit


def test_explain_dispatch_prints_full_chain(decode_registry, npu_hardware):
    text = explain_dispatch(_SIG_INT8, npu_hardware, registry=decode_registry)
    assert "cann/attention/decode_aclnn_v1" in text
    assert "[reject] ascendc/attention/decode_v1" in text
    assert "确定性启发式" in text


def _external_descriptor(compile_mode=CompileMode.AOT) -> KernelDescriptor:
    return KernelDescriptor(
        kernel_id="vllm-ascend/attention/decode_v1",
        backend=Backend.EXTERNAL,
        capability=CapabilitySpec(op="attention.decode"),
        compile_mode=compile_mode,
    )


@pytest.mark.parametrize("allow_custom", [None, "0", "1"])
@pytest.mark.parametrize("allow_reference", [False, True])
@pytest.mark.parametrize("compile_mode", [CompileMode.AOT, CompileMode.JIT])
def test_auto_visibility_by_kernel_origin(
    decode_registry, npu_hardware, monkeypatch,
    allow_custom, allow_reference, compile_mode,
):
    monkeypatch.delenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", raising=False)
    monkeypatch.delenv("FLASHINFER_NPU_JIT_DISABLE", raising=False)
    if allow_custom is not None:
        monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", allow_custom)
    decode_registry.freeze()
    external = _external_descriptor(compile_mode)
    custom = _custom_descriptor()
    decode_registry.register_external(external)
    decode_registry.register_custom(custom)

    decision = dispatch(
        _SIG_FP16, npu_hardware, registry=decode_registry,
        allow_reference=allow_reference,
    )
    assert {r.kernel_id: r.accepted for r in decision.reports} == {
        "ascendc/attention/decode_v1": True,
        "cann/attention/decode_aclnn_v1": True,
        "triton/attention/decode_v1": True,
        "reference/attention/decode_v1": allow_reference,
        external.kernel_id: True,
        custom.kernel_id: allow_custom == "1",
    }
    assert external.kernel_id in (decision.kernel_id, *decision.fallback_chain)


@pytest.mark.parametrize("compile_mode", [CompileMode.AOT, CompileMode.JIT])
def test_external_kernel_selected_without_custom_opt_in(
    npu_hardware, monkeypatch, compile_mode,
):
    monkeypatch.delenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", raising=False)
    monkeypatch.delenv("FLASHINFER_NPU_JIT_DISABLE", raising=False)
    registry = KernelRegistry()
    descriptor = _external_descriptor(compile_mode)
    registry.register_external(descriptor)
    decision = dispatch(_SIG_FP16, npu_hardware, registry=registry)
    assert decision.kernel_id == descriptor.kernel_id
    assert decision.backend == Backend.EXTERNAL


def test_external_backend_policy_only_accepts_external(
    decode_registry, npu_hardware, monkeypatch,
):
    monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "0")
    external = _external_descriptor()
    decode_registry.register_external(external)
    decode_registry.register_custom(_custom_descriptor())
    decision = dispatch(
        _SIG_FP16, npu_hardware, backend="external", registry=decode_registry,
    )
    assert decision.kernel_id == external.kernel_id
    assert {r.kernel_id for r in decision.reports if r.accepted} == {external.kernel_id}


@pytest.mark.parametrize("restriction, reason", [
    ("dtype", "dtype"), ("target_soc", "target SoC"), ("jit", "JIT 已禁用"),
])
def test_external_still_obeys_compatibility_filters(
    decode_registry, npu_hardware, monkeypatch, restriction, reason,
):
    external = _external_descriptor()
    if restriction == "dtype":
        external = replace(external, capability=CapabilitySpec(
            op="attention.decode", dtypes=("bfloat16",),
        ))
    elif restriction == "target_soc":
        external = replace(external, target_soc="Ascend950")
    else:
        external = replace(external, compile_mode=CompileMode.JIT)
        monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    decode_registry.register_external(external)
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    report = next(r for r in decision.reports if r.kernel_id == external.kernel_id)
    assert not report.accepted
    assert any(reason in detail for detail in report.reasons)


def test_external_registration_invalidates_cached_decision(
    decode_registry, npu_hardware,
):
    dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert dispatch(_SIG_FP16, npu_hardware, registry=decode_registry).cache_hit
    revision = decode_registry.revision
    decode_registry.freeze()
    external = _external_descriptor()
    decode_registry.register_external(external)
    assert decode_registry.revision == revision + 1
    decision = dispatch(_SIG_FP16, npu_hardware, registry=decode_registry)
    assert not decision.cache_hit
    assert any(r.kernel_id == external.kernel_id and r.accepted for r in decision.reports)


@pytest.mark.parametrize("entry, backend", [
    ("register", Backend.EXTERNAL),
    ("register", Backend.CUSTOM),
    ("register_custom", Backend.EXTERNAL),
    ("register_custom", Backend.CANN),
    ("register_external", Backend.CUSTOM),
    ("register_external", Backend.CANN),
])
def test_registration_entries_reject_other_origins(entry, backend):
    registry = KernelRegistry()
    descriptor = replace(
        _external_descriptor(), backend=backend,
        kernel_id="custom/attention/decode_v1" if backend == Backend.CUSTOM
        else "vllm-ascend/attention/decode_v1",
    )
    with pytest.raises(ValueError):
        getattr(registry, entry)(descriptor)
    assert registry.descriptors() == ()
    assert registry.revision == 0


@pytest.mark.parametrize("namespace", ["vllm-ascend", "mindspeed", "atb"])
def test_external_uses_repository_namespace(namespace):
    registry = KernelRegistry()
    descriptor = replace(_external_descriptor(), kernel_id=f"{namespace}/attention/decode_v1")
    registry.register_external(descriptor)
    assert registry.get(descriptor.kernel_id) is descriptor


@pytest.mark.parametrize("kernel_id", [
    "decode_v1", "/attention/decode_v1", "vllm-ascend/",
    *(f"{backend.value}/attention/decode_v1" for backend in Backend),
])
def test_external_rejects_missing_or_reserved_namespace(kernel_id):
    registry = KernelRegistry()
    with pytest.raises(ValueError, match="前缀"):
        registry.register_external(replace(_external_descriptor(), kernel_id=kernel_id))
    assert registry.revision == 0


@pytest.mark.parametrize("backend", [
    Backend.CANN, Backend.ASCENDC, Backend.TRITON, Backend.REFERENCE,
])
def test_custom_namespace_is_reserved_from_builtins(backend):
    registry = KernelRegistry()
    with pytest.raises(ValueError, match="custom/"):
        registry.register(replace(_custom_descriptor(), backend=backend))
    assert registry.revision == 0


def test_duplicate_external_registration_cannot_replace_existing():
    registry = KernelRegistry()
    descriptor = _external_descriptor()
    registry.register_external(descriptor)
    with pytest.raises(ValueError, match="duplicate kernel_id"):
        registry.register_external(replace(descriptor, priority=99))
    assert registry.get(descriptor.kernel_id) is descriptor
    assert registry.revision == 1


def test_global_external_registration_is_exported_and_works_after_freeze(monkeypatch):
    import flashinfer
    import flashinfer.runtime as runtime
    from flashinfer.runtime import registry as registry_module

    registry = KernelRegistry()
    registry.freeze()
    monkeypatch.setattr(registry_module, "_GLOBAL_REGISTRY", registry)
    assert flashinfer.register_external_kernel is runtime.register_external_kernel
    descriptor = _external_descriptor()
    runtime.register_external_kernel(descriptor)
    assert registry.get(descriptor.kernel_id) is descriptor
    assert registry.frozen


@pytest.mark.parametrize("entry, descriptor", [
    ("register_custom_kernel", _external_descriptor()),
    ("register_external_kernel", _custom_descriptor()),
])
def test_global_registration_entries_reject_other_origins(monkeypatch, entry, descriptor):
    import flashinfer.runtime as runtime
    from flashinfer.runtime import registry as registry_module

    registry = KernelRegistry()
    registry.freeze()
    monkeypatch.setattr(registry_module, "_GLOBAL_REGISTRY", registry)
    with pytest.raises(ValueError, match="backend 必须为"):
        getattr(runtime, entry)(descriptor)
    assert registry.descriptors() == ()
    assert registry.revision == 0


def test_external_identity_survives_json_round_trip():
    from flashinfer.runtime.manifest import descriptor_from_dict

    descriptor = _external_descriptor()
    payload = json.loads(json.dumps(descriptor.to_dict()))
    assert payload["backend"] == "external"
    restored = descriptor_from_dict(payload)
    assert restored.backend == Backend.EXTERNAL
    assert restored.fingerprint == descriptor.fingerprint
