"""T7：可配置档位、全部 backend、tuning 路径与确定性。"""

from dataclasses import replace
import importlib
from itertools import permutations
from unittest.mock import Mock

import pytest

from flashinfer import config
from flashinfer.runtime.dispatch import clear_dispatch_cache, dispatch, explain_dispatch
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import (
    ArtifactFormat, ArtifactKind, ArtifactRef, Backend, CapabilitySpec, CompileMode,
    DispatchError, KernelArgumentABI, KernelArgumentDirection, KernelArgumentPassing,
    KernelBinaryABI, KernelDescriptor, KernelErrorABI, KernelErrorCodeABI,
    KernelLaunchABI, ShapeRange,
)
from flashinfer.runtime.signature import InputSignature
from flashinfer.tuning import query_tuning_record


_DISPATCH = importlib.import_module("flashinfer.runtime.dispatch")
_ENV = "FLASHINFER_NPU_DISPATCH_ORDER"
_GROUPS = ("external", "cann", "aot", "jit", "reference")
_SIG = InputSignature("attention.decode", dtypes=("float16",))
_CANN = "cann/attention/decode_aclnn_v1"
_AOT = "ascendc/attention/decode_v1"
_JIT = "triton/attention/decode_v1"
_REFERENCE = "reference/attention/decode_v1"


@pytest.fixture(autouse=True)
def _default_order(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)


def _descriptor(backend, mode=CompileMode.NONE, *, priority=0, suffix="v1"):
    namespace = "vllm-ascend" if backend == Backend.EXTERNAL else backend.value
    extras = {}
    formats = {
        Backend.ASCENDC_AOT: (ArtifactKind.FILE, ArtifactFormat.ASCENDC_OBJECT),
        Backend.ASCENDC_JIT: (ArtifactKind.JIT_SOURCE, ArtifactFormat.ASCENDC_SOURCE),
        Backend.ACLNN: (ArtifactKind.BUILTIN, ArtifactFormat.ACLNN_BUILTIN),
    }
    if backend in formats:
        kind, format_ = formats[backend]
        extras = dict(
            artifact=ArtifactRef(
                kind, format_, "test/kernel", "0" * 64, "Ascend910B", "test",
                size_bytes=None if kind == ArtifactKind.BUILTIN else 0,
            ),
            launch_abi=KernelLaunchABI("test.v1", "entry", ("stream",)),
            binary_abi=KernelBinaryABI(
                "test.binary.v1",
                (KernelArgumentABI("stream", KernelArgumentPassing.OPAQUE_HANDLE,
                                   KernelArgumentDirection.INPUT),),
                KernelErrorABI("test.error.v1", (KernelErrorCodeABI("success", 0),)),
            ),
        )
    return KernelDescriptor(
        f"{namespace}/ordering/{suffix}", backend=backend, compile_mode=mode,
        priority=priority, capability=CapabilitySpec(_SIG.op), **extras,
    )


def _registry(descriptors):
    registry = KernelRegistry()
    for descriptor in descriptors:
        if descriptor.backend == Backend.EXTERNAL:
            registry.register_external(descriptor)
        elif descriptor.backend == Backend.CUSTOM:
            registry.register_custom(descriptor)
        else:
            registry.register(descriptor)
    return registry


@pytest.mark.parametrize("backend, expected", [
    (Backend.CANN, (1, 1, 1)),
    (Backend.ASCENDC, (2, 3, 3)),
    (Backend.TRITON, (2, 3, 3)),
    (Backend.EXTERNAL, (0, 0, 0)),
    (Backend.CUSTOM, (2, 3, 3)),
    (Backend.REFERENCE, (4, 4, 4)),
    (Backend.ASCENDC_AOT, (2, 2, 2)),
    (Backend.ASCENDC_JIT, (3, 3, 3)),
    (Backend.ACLNN, (1, 1, 1)),
])
def test_every_backend_and_compile_mode_has_an_explicit_tier(backend, expected):
    assert set(_DISPATCH._BACKEND_GROUPS) == set(Backend)
    assert set(_DISPATCH._COMPILE_GROUPS) == set(CompileMode)
    modes = (CompileMode.AOT, CompileMode.JIT, CompileMode.NONE)
    for mode, tier in zip(modes, expected):
        assert _DISPATCH._heuristic_tier(_descriptor(backend, mode)) == tier


def test_default_tiers_cover_legacy_backends_and_outrank_priority(npu_hardware, monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "1")
    # 按期望顺序列出；后面档位给更高 priority，以验证 priority 不跨档。
    expected = (
        _descriptor(Backend.EXTERNAL, CompileMode.NONE, priority=-10),
        _descriptor(Backend.ACLNN, priority=100),
        _descriptor(Backend.CANN, priority=100),
        _descriptor(Backend.ASCENDC, CompileMode.AOT, priority=1000, suffix="aot"),
        _descriptor(Backend.ASCENDC_AOT, priority=1000),
        _descriptor(Backend.CUSTOM, CompileMode.AOT, priority=1000, suffix="aot"),
        _descriptor(Backend.TRITON, CompileMode.AOT, priority=1000, suffix="aot"),
        _descriptor(Backend.ASCENDC, CompileMode.JIT, priority=10000, suffix="jit"),
        _descriptor(Backend.ASCENDC_JIT, priority=10000),
        _descriptor(Backend.CUSTOM, CompileMode.JIT, priority=10000, suffix="jit"),
        _descriptor(Backend.TRITON, CompileMode.JIT, priority=10000, suffix="jit"),
        _descriptor(Backend.REFERENCE, priority=100000),
    )
    for registration in (expected, tuple(reversed(expected))):
        clear_dispatch_cache()
        decision = dispatch(_SIG, npu_hardware, registry=_registry(registration),
                            allow_reference=True)
        assert (decision.kernel_id, *decision.fallback_chain) == tuple(
            d.kernel_id for d in expected
        )
        assert decision.reason == "确定性启发式 tier 0（已验证外部仓实现）"


@pytest.mark.parametrize("mode", list(CompileMode))
def test_external_keeps_its_tier_independent_of_compile_mode(
    decode_registry, npu_hardware, mode,
):
    external = _descriptor(Backend.EXTERNAL, mode, priority=-100)
    decode_registry.register_external(external)
    decision = dispatch(_SIG, npu_hardware, registry=decode_registry)
    assert decision.kernel_id == external.kernel_id
    assert decision.fallback_chain == (_CANN, _AOT, _JIT)


def test_every_configured_order_controls_champion_and_fallbacks(npu_hardware, monkeypatch):
    descriptors = (
        _descriptor(Backend.EXTERNAL), _descriptor(Backend.CANN),
        _descriptor(Backend.ASCENDC, CompileMode.AOT),
        _descriptor(Backend.TRITON, CompileMode.JIT), _descriptor(Backend.REFERENCE),
    )
    by_group = dict(zip(_GROUPS, (d.kernel_id for d in descriptors)))
    registry = _registry(descriptors)
    for order in permutations(_GROUPS):
        monkeypatch.setenv(_ENV, ",".join(order))
        assert config.dispatch_order() == order
        decision = dispatch(_SIG, npu_hardware, registry=registry, allow_reference=True)
        assert (decision.kernel_id, *decision.fallback_chain) == tuple(
            by_group[group] for group in order
        )
        assert not decision.cache_hit
        assert decision.reason.startswith("确定性启发式 tier 0（")
    # 把 reference 提前并不能绕过 opt-in。
    decision = dispatch(_SIG, npu_hardware, registry=registry)
    assert by_group["reference"] not in (decision.kernel_id, *decision.fallback_chain)


@pytest.mark.parametrize("raw, detail", [
    ("", "缺少档位"),
    ("external,cann,aot,jit", "缺少档位 ['reference']"),
    ("external,cann,aot,jit,reference,cann", "重复档位 ['cann']"),
    ("external,cann,aot,jit,other", "未知档位 ['other']"),
    ("external,cann,aot,jit,reference,", "未知档位 ['']"),
    ("EXTERNAL,cann,aot,jit,reference", "未知档位 ['EXTERNAL']"),
    ("external;cann;aot;jit;reference", "未知档位"),
    ("auto", "未知档位 ['auto']"),
])
def test_invalid_order_reports_variable_value_and_problem(monkeypatch, raw, detail):
    monkeypatch.setenv(_ENV, raw)
    with pytest.raises(ValueError) as error:
        config.dispatch_order()
    message = str(error.value)
    assert _ENV in message and repr(raw) in message and detail in message
    assert "每档恰好一次" in message


def test_order_change_partitions_decision_cache(decode_registry, npu_hardware, monkeypatch):
    first = dispatch(_SIG, npu_hardware, registry=decode_registry)
    assert first.kernel_id == _CANN and not first.cache_hit
    assert dispatch(_SIG, npu_hardware, registry=decode_registry).cache_hit
    monkeypatch.setenv(_ENV, "aot,external,cann,jit,reference")
    second = dispatch(_SIG, npu_hardware, registry=decode_registry)
    assert second.kernel_id == _AOT and not second.cache_hit
    assert second.reason == "确定性启发式 tier 0（自研 AOT）"
    assert dispatch(_SIG, npu_hardware, registry=decode_registry).cache_hit
    monkeypatch.delenv(_ENV)
    restored = dispatch(_SIG, npu_hardware, registry=decode_registry)
    assert restored.kernel_id == _CANN and restored.cache_hit


def test_whitespace_is_normalized_in_cache_key(decode_registry, npu_hardware, monkeypatch):
    dispatch(_SIG, npu_hardware, registry=decode_registry)
    monkeypatch.setenv(_ENV, " external , cann , aot , jit , reference ")
    assert config.dispatch_order() == _GROUPS
    assert dispatch(_SIG, npu_hardware, registry=decode_registry).cache_hit


@pytest.mark.parametrize("disable_cache", [False, True])
def test_invalid_order_is_checked_before_using_cache(
    decode_registry, npu_hardware, monkeypatch, disable_cache,
):
    dispatch(_SIG, npu_hardware, registry=decode_registry)
    if disable_cache:
        monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1")
    monkeypatch.setenv(_ENV, "cann")
    with pytest.raises(ValueError, match=_ENV):
        dispatch(_SIG, npu_hardware, registry=decode_registry)


def test_config_diagnostics_include_effective_order(monkeypatch):
    monkeypatch.setenv(_ENV, "jit,aot,cann,external,reference")
    assert "dispatch_order: jit,aot,cann,external,reference" in config.format_config()


def test_real_tuning_stub_miss_preserves_heuristic_decision(
    decode_registry, npu_hardware, monkeypatch,
):
    assert query_tuning_record(_SIG, npu_hardware) is None
    before = dispatch(_SIG, npu_hardware, registry=decode_registry)
    clear_dispatch_cache()
    query = Mock(wraps=query_tuning_record)
    monkeypatch.setattr(_DISPATCH, "query_tuning_record", query)
    after = dispatch(_SIG, npu_hardware, registry=decode_registry)
    query.assert_called_once_with(_SIG, npu_hardware, registry=decode_registry)
    assert after == before
    assert after.reason == "确定性启发式 tier 1（CANN / ACLNN 算子）"


@pytest.mark.parametrize("order", [_GROUPS, tuple(reversed(_GROUPS))])
def test_tuning_champion_overrides_order_and_explains_source(
    decode_registry, npu_hardware, monkeypatch, order,
):
    monkeypatch.setenv(_ENV, ",".join(order))
    query = Mock(return_value=_JIT)
    monkeypatch.setattr(_DISPATCH, "query_tuning_record", query)
    decision = dispatch(_SIG, npu_hardware, registry=decode_registry)
    assert decision.kernel_id == _JIT
    expected_fallbacks = (_CANN, _AOT) if order == _GROUPS else (_AOT, _CANN)
    assert decision.fallback_chain == expected_fallbacks
    assert decision.reason == "实测 tuning 记录命中：kernel_id=triton/attention/decode_v1"
    again = dispatch(_SIG, npu_hardware, registry=decode_registry)
    assert again.cache_hit and again.reason == decision.reason
    assert decision.reason in explain_dispatch(_SIG, npu_hardware, registry=decode_registry)
    query.assert_called_once_with(_SIG, npu_hardware, registry=decode_registry)


@pytest.mark.parametrize("restriction", [
    "unknown", "dtype", "policy", "reference", "custom", "jit",
    "target_soc", "shape", "physical_layout",
])
def test_tuning_cannot_select_an_unavailable_candidate(
    decode_registry, npu_hardware, monkeypatch, restriction,
):
    signature = _SIG
    backend = "auto"
    tuned_id = "missing/kernel"
    if restriction == "dtype":
        signature = replace(signature, dtypes=("int8",))
        tuned_id = _AOT
    elif restriction == "policy":
        backend, tuned_id = "cann", _JIT
    elif restriction == "reference":
        tuned_id = _REFERENCE
    elif restriction == "custom":
        monkeypatch.setenv("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "0")
        custom = _descriptor(Backend.CUSTOM, CompileMode.AOT)
        decode_registry.register_custom(custom)
        tuned_id = custom.kernel_id
    elif restriction == "jit":
        monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
        tuned_id = _JIT
    elif restriction in ("target_soc", "shape", "physical_layout"):
        external = _descriptor(Backend.EXTERNAL)
        if restriction == "target_soc":
            external = replace(external, target_soc="Ascend950")
        elif restriction == "shape":
            external = replace(external, capability=CapabilitySpec(
                _SIG.op, shape_ranges=(ShapeRange("head_dim", 0, -1, 64, 128),),
            ))
            signature = replace(signature, shapes=((256,),))
        else:
            external = replace(external, capability=CapabilitySpec(
                _SIG.op, physical_layouts=("ND",),
            ))
            signature = replace(signature, physical_layouts=("NZ",))
        decode_registry.register_external(external)
        tuned_id = external.kernel_id
    query = Mock(return_value=tuned_id)
    monkeypatch.setattr(_DISPATCH, "query_tuning_record", query)
    decision = dispatch(signature, npu_hardware, backend=backend, registry=decode_registry)
    assert decision.kernel_id == _CANN
    assert decision.reason == (
        "确定性启发式 tier 1（CANN / ACLNN 算子）；"
        f"tuning 记录 {tuned_id} 不在当前可用候选中"
    )
    assert tuned_id not in (decision.kernel_id, *decision.fallback_chain)
    if restriction != "unknown":
        report = next(r for r in decision.reports if r.kernel_id == tuned_id)
        assert not report.accepted and report.reasons
    query.assert_called_once_with(signature, npu_hardware, registry=decode_registry)


def test_explicit_kernel_id_takes_precedence_over_tuning(
    decode_registry, npu_hardware, monkeypatch,
):
    query = Mock(return_value=_JIT)
    monkeypatch.setattr(_DISPATCH, "query_tuning_record", query)
    decision = dispatch(_SIG, npu_hardware, kernel_id=_AOT, registry=decode_registry)
    assert decision.kernel_id == _AOT and decision.reason == "显式 kernel_id 锁定"
    query.assert_not_called()


def test_no_survivors_does_not_query_tuning(decode_registry, npu_hardware, monkeypatch):
    query = Mock(return_value=_CANN)
    monkeypatch.setattr(_DISPATCH, "query_tuning_record", query)
    with pytest.raises(DispatchError, match="无可用 kernel"):
        dispatch(InputSignature("unknown.op"), npu_hardware, registry=decode_registry)
    query.assert_not_called()


def test_tie_break_is_priority_then_kernel_id_for_every_registration_order(
    npu_hardware, monkeypatch,
):
    monkeypatch.setenv("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "1")
    descriptors = (
        _descriptor(Backend.ASCENDC, CompileMode.AOT, priority=2, suffix="z"),
        _descriptor(Backend.ASCENDC, CompileMode.AOT, priority=2, suffix="a"),
        _descriptor(Backend.ASCENDC, CompileMode.AOT, priority=1, suffix="0"),
    )
    expected = tuple(descriptors[i].kernel_id for i in (1, 0, 2))
    for registration in permutations(descriptors):
        registry = _registry(registration)
        first = dispatch(_SIG, npu_hardware, registry=registry)
        second = dispatch(_SIG, npu_hardware, registry=registry)
        assert first == second
        assert (first.kernel_id, *first.fallback_chain) == expected
