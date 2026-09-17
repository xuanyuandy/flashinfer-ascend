"""dispatcher 行为（§5.3 / §5.4.3 / §10.5 / §12，全部 host 可测）。"""

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
