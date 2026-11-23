"""T19A: real registry partition, source identities and both decision levels."""

from tests.production_inventory import DEFAULT_DESCRIPTOR_COUNT, PRODUCTION_KERNEL_IDS, STACK_COUNTS

from collections import Counter
from dataclasses import replace
import importlib
from itertools import permutations
import json

import pytest

from flashinfer.runtime import (
    ArtifactFormat, ArtifactKind, ArtifactRef, Backend, CapabilitySpec, CompileMode,
    DispatchError, HardwareSignature, ImplementationStack, InputSignature,
    KernelArgumentABI, KernelArgumentDirection, KernelArgumentPassing, KernelBinaryABI,
    KernelDescriptor, KernelErrorABI, KernelErrorCodeABI, KernelLaunchABI, KernelRegistry,
    ShapeRange, StackReport, clear_dispatch_cache, dispatch, explain_dispatch, get_global_registry,
)
from flashinfer.runtime.manifest import descriptor_from_dict


HW = HardwareSignature("Ascend950", 32)
SIG = InputSignature("test.op", dtypes=("float32",), shapes=((4, 128),))
DISPATCH = importlib.import_module("flashinfer.runtime.dispatch")


@pytest.fixture(autouse=True)
def default_policies(monkeypatch):
    for suffix in ("DISPATCH_ORDER", "ALLOW_CUSTOM_KERNELS", "JIT_DISABLE", "DISABLE_DISPATCH_CACHE"):
        monkeypatch.delenv("FLASHINFER_NPU_" + suffix, raising=False)


def descriptor(kernel_id, backend=Backend.TRITON, **kwargs):
    return KernelDescriptor(kernel_id, "test.op", backend, **kwargs)


def registry_of(*descriptors):
    registry = KernelRegistry()
    for d in descriptors:
        if d.backend == Backend.EXTERNAL:
            registry.register_external(d)
        elif d.backend == Backend.CUSTOM:
            registry.register_custom(d)
        else:
            registry.register(d)
    return registry


def stack_report(decision, stack_id):
    return next(s for s in decision.stack_reports if s.stack_id == stack_id)


def test_builtin_partition_is_complete_disjoint_and_has_no_within_stack_data():
    registry = get_global_registry()
    before = [(d.kernel_id, d.fingerprint) for d in registry.descriptors()]
    stacks = registry.stacks()
    assert {s.stack_id: len(s.descriptors) for s in stacks} == STACK_COUNTS
    assert all(isinstance(s, ImplementationStack) for s in stacks)
    flat = [d for s in stacks for d in s.descriptors]
    assert len(flat) == len({d.kernel_id for d in flat}) == DEFAULT_DESCRIPTOR_COUNT
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID
    from flashinfer.runtime.registry import _builtin_descriptors

    legacy = _builtin_descriptors()
    assert len(legacy) == 81
    assert {d.kernel_id for d in flat} - {d.kernel_id for d in legacy} == PRODUCTION_KERNEL_IDS
    assert {id(d) for d in flat} == {id(d) for d in registry.descriptors()}
    duplicates = {
        (s.stack_id, op): frozenset(d.kernel_id for d in s.descriptors if d.op == op)
        for s in stacks for op, count in Counter(d.op for d in s.descriptors).items()
        if count > 1
    }
    assert duplicates == {
        ("ascendc", "ascendc.single_prefill_catlass"): frozenset({
            "ascendc/t4/flash_attention/catlass/op_package/Ascend910B",
            "ascendc/t4/flash_attention/catlass/op_package/Ascend910C",
            "ascendc/t4/flash_attention/catlass/op_package/Ascend950",
        }),
        ("ascendc", "gemm.grouped_matmul_swiglu_quant"): frozenset({
            "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B",
            "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910C",
        }),
        ("ascendc", "attention.sparse_flash_attention"): frozenset({
            "ascendc/attention/sparse_flash_attention/native/Ascend910B",
            "ascendc/attention/sparse_flash_attention/native/Ascend910C",
        }),
        ("ascendc", "attention.dsa_sparse_prolog"): frozenset({
            "ascendc/attention/dsa_sparse_prolog/native/Ascend910B",
            "ascendc/attention/dsa_sparse_prolog/native/Ascend910C",
        }),
    }
    assert before == [(d.kernel_id, d.fingerprint) for d in registry.descriptors()]
    assert registry.stacks(op="missing.op") == ()


def test_stack_identity_is_stable_across_ops_and_compile_modes():
    a = descriptor("ascendc/a", Backend.ASCENDC, compile_mode=CompileMode.AOT)
    b = replace(a, kernel_id="ascendc/b", op="other.op", compile_mode=CompileMode.JIT)
    registry = registry_of(b, a)
    stack, = registry.stacks()
    assert stack.stack_id == "ascendc" and stack.descriptors == (a, b)
    assert registry.stacks("other.op") == (ImplementationStack("ascendc", (b,)),)
    assert registry.stack_id_for(b.kernel_id) == registry.stack_id_for(a.kernel_id)
    with pytest.raises(KeyError, match="未注册"):
        registry.stack_id_for("unknown")


@pytest.mark.parametrize("backend,stack_id,kind,format_", [
    (Backend.ACLNN, "cann", ArtifactKind.BUILTIN, ArtifactFormat.ACLNN_BUILTIN),
    (Backend.ASCENDC_AOT, "ascendc", ArtifactKind.FILE, ArtifactFormat.ASCENDC_OBJECT),
    (Backend.ASCENDC_JIT, "ascendc", ArtifactKind.JIT_SOURCE, ArtifactFormat.ASCENDC_SOURCE),
])
def test_legacy_backend_aliases_share_the_source_stack(backend, stack_id, kind, format_):
    legacy = descriptor(
        "legacy/kernel", backend,
        artifact=ArtifactRef(kind, format_, "test/kernel", "0" * 64, "Ascend950", "test",
                             size_bytes=None if kind == ArtifactKind.BUILTIN else 0),
        launch_abi=KernelLaunchABI("test.v1", "entry", ("stream",)),
        binary_abi=KernelBinaryABI(
            "test.binary.v1",
            (KernelArgumentABI("stream", KernelArgumentPassing.OPAQUE_HANDLE,
                               KernelArgumentDirection.INPUT),),
            KernelErrorABI("test.error.v1", (KernelErrorCodeABI("success", 0),)),
        ),
    )
    modern = descriptor(stack_id + "/modern", Backend(stack_id))
    registry = registry_of(legacy, modern)
    stack, = registry.stacks()
    assert stack.stack_id == stack_id
    assert {d.kernel_id for d in stack.descriptors} == {legacy.kernel_id, modern.kernel_id}


def test_external_repository_namespaces_are_separate_stacks_after_round_trip():
    a = descriptor("repo-a/gemma/block128", Backend.EXTERNAL)
    b = descriptor("repo-a/gemma/block256", Backend.EXTERNAL)
    c = descriptor("repo-b/gemma/block128", Backend.EXTERNAL, priority=1)
    restored = [descriptor_from_dict(json.loads(json.dumps(d.to_dict()))) for d in (a, b, c)]
    registry = registry_of(*restored)
    assert {s.stack_id: [d.kernel_id for d in s.descriptors] for s in registry.stacks()} == {
        "external:repo-a": [a.kernel_id, b.kernel_id],
        "external:repo-b": [c.kernel_id],
    }
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.stack_id == "external:repo-b" and decision.kernel_id == c.kernel_id
    a_report = stack_report(decision, "external:repo-a")
    assert a_report.accepted and not a_report.selected
    assert "priority 1 > 0" in ";".join(a_report.reasons)
    assert a_report.representative_kernel_id == a.kernel_id
    assert "栈内排序落后" in ";".join(a_report.implementations[1].reasons)


def test_frozen_registry_views_and_cache_see_new_external_and_custom_records():
    registry = registry_of(descriptor("triton/a"))
    snapshot = registry.stacks()
    registry.freeze()
    assert dispatch(SIG, HW, registry=registry).stack_id == "triton"
    assert dispatch(SIG, HW, registry=registry).cache_hit
    registry.register_external(descriptor("repo-a/a", Backend.EXTERNAL))
    registry.register_custom(descriptor("custom/a", Backend.CUSTOM))
    assert [s.stack_id for s in snapshot] == ["triton"]
    assert [s.stack_id for s in registry.stacks()] == ["custom", "external:repo-a", "triton"]
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.stack_id == "external:repo-a" and not decision.cache_hit
    custom = stack_report(decision, "custom")
    assert not custom.accepted and not custom.selected
    assert "默认不进 auto" in custom.implementations[0].reasons[0]


def test_registration_during_filtering_does_not_mix_registry_snapshots(monkeypatch):
    registry = registry_of(descriptor("triton/a"))
    registry.freeze()
    external = descriptor("repo-a/new", Backend.EXTERNAL)
    predicate = CapabilitySpec.unsupported_reasons
    registered = False

    def register_while_filtering(capability, signature, hardware):
        nonlocal registered
        if not registered:
            registered = True
            registry.register_external(external)
        return predicate(capability, signature, hardware)

    monkeypatch.setattr(CapabilitySpec, "unsupported_reasons", register_while_filtering)
    first = dispatch(SIG, HW, registry=registry)
    assert first.kernel_id == "triton/a"
    assert [s.stack_id for s in first.stack_reports] == ["triton"]
    second = dispatch(SIG, HW, registry=registry)
    assert not second.cache_hit and second.kernel_id == external.kernel_id
    assert {s.stack_id for s in second.stack_reports} == {"triton", "external:repo-a"}


def test_explanation_separates_eligible_losing_stack_and_filtered_stack():
    registry = registry_of(descriptor("cann/a", Backend.CANN), descriptor("triton/a"),
                           descriptor("reference/a", Backend.REFERENCE))
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.stack_id == "cann"
    assert all(isinstance(s, StackReport) for s in decision.stack_reports)
    triton = stack_report(decision, "triton")
    assert triton.accepted and not triton.selected
    assert "tier 1 < 3" in ";".join(triton.reasons)
    assert not stack_report(decision, "reference").accepted
    # Preserve the old filter-only data instead of changing accept into select.
    assert [(r.kernel_id, r.accepted, bool(r.reasons)) for r in decision.reports] == [
        ("cann/a", True, False), ("reference/a", False, True), ("triton/a", True, False),
    ]
    text = explain_dispatch(SIG, HW, registry=registry)
    assert "第一级：选择实现栈" in text and "第二级：各栈内的实现" in text
    assert "[select] cann" in text and "[accept] triton" in text and "[reject] reference" in text
    assert "栈可用但未选中" in text and "reference 需显式 opt-in" in text
    assert "栈内唯一通过过滤的实现" in text


@pytest.mark.parametrize("dimension", ["tier", "priority", "kernel_id"])
def test_within_stack_selection_explains_each_existing_tie_break(dimension):
    first = descriptor("triton/a", compile_mode=CompileMode.JIT)
    second = descriptor("triton/b", compile_mode=CompileMode.JIT)
    if dimension == "tier":
        first = replace(first, compile_mode=CompileMode.AOT, priority=-100)
    elif dimension == "priority":
        first = replace(first, priority=10)
    expected = {"tier": "tier 2 < 3", "priority": "priority 10 > 0",
                "kernel_id": "kernel_id 字典序"}[dimension]
    for insertion in permutations((first, second)):
        clear_dispatch_cache()
        decision = dispatch(SIG, HW, registry=registry_of(*insertion))
        assert decision.kernel_id == first.kernel_id and decision.stack_id == "triton"
        stack, = decision.stack_reports
        assert stack.representative_kernel_id == first.kernel_id
        loser = next(r for r in stack.implementations if r.kernel_id == second.kernel_id)
        assert loser.accepted and expected in ";".join(loser.reasons)
        assert decision.fallback_chain == (second.kernel_id,)


def test_stack_frontier_excludes_filtered_configs_and_preserves_interleaved_fallbacks():
    aot = descriptor("ascendc/aot", Backend.ASCENDC, compile_mode=CompileMode.AOT)
    jit = descriptor("ascendc/jit", Backend.ASCENDC, compile_mode=CompileMode.JIT)
    triton = descriptor("triton/aot", compile_mode=CompileMode.AOT)
    registry = registry_of(jit, triton, aot)
    first = dispatch(SIG, HW, registry=registry)
    assert (first.kernel_id, *first.fallback_chain) == (aot.kernel_id, triton.kernel_id, jit.kernel_id)
    # Best declared config is now ineligible; stack ranking must use the JIT
    # frontier rather than the source's former AOT tier.
    rejected = replace(aot, capability=CapabilitySpec(SIG.op, dtypes=("int8",)))
    clear_dispatch_cache()
    second = dispatch(SIG, HW, registry=registry_of(jit, triton, rejected))
    assert second.stack_id == "triton" and second.kernel_id == triton.kernel_id
    ascendc = stack_report(second, "ascendc")
    assert ascendc.accepted and ascendc.representative_kernel_id == jit.kernel_id
    assert "tier 2 < 3" in ";".join(ascendc.reasons)
    assert not ascendc.implementations[0].accepted


def test_all_filter_reasons_are_retained_in_the_second_level(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    bad = descriptor(
        "triton/bad", compile_mode=CompileMode.JIT, target_soc="Ascend910B",
        capability=CapabilitySpec(SIG.op, dtypes=("int8",), layouts=("HND",),
                                  soc_models=("Ascend910B",), min_aicore=100,
                                  shape_ranges=(ShapeRange("hidden", 0, -1, 1, 64),)),
    )
    signature = replace(SIG, layouts=("NHD",))
    registry = registry_of(bad, descriptor("cann/ok", Backend.CANN))
    decision = dispatch(signature, HW, registry=registry)
    original = next(r for r in decision.reports if r.kernel_id == bad.kernel_id)
    nested, = stack_report(decision, "triton").implementations
    assert nested == original
    stack_reasons = ";".join(stack_report(decision, "triton").reasons)
    text = explain_dispatch(signature, HW, registry=registry)
    for reason in original.reasons:
        assert reason in text
        assert reason in stack_reasons
    assert len(original.reasons) == 7


def test_no_candidate_error_explains_both_levels_and_every_rejection():
    registry = registry_of(
        descriptor("reference/a", Backend.REFERENCE),
        descriptor("triton/a", capability=CapabilitySpec(SIG.op, dtypes=("int8",))),
    )
    with pytest.raises(DispatchError) as caught:
        explain_dispatch(SIG, HW, registry=registry)
    text = str(caught.value)
    for fragment in ("第一级", "第二级", "[reject] reference", "[reject] triton",
                     "reference 需显式 opt-in", "dtype float32", "栈内全部 1 个实现被过滤"):
        assert fragment in text


def test_no_registered_op_remains_an_error_with_empty_two_level_trace():
    with pytest.raises(DispatchError) as caught:
        dispatch(SIG, HW, registry=KernelRegistry())
    text = str(caught.value)
    assert "第一级" in text and "第二级" in text and "无该 op 的任何 kernel" in text


def test_explicit_kernel_override_explains_evaluation_scope_and_preserves_bypass(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    chosen = descriptor("triton/a", compile_mode=CompileMode.JIT)
    registry = registry_of(chosen, descriptor("cann/a", Backend.CANN))
    decision = dispatch(SIG, HW, registry=registry, backend="cann", kernel_id=chosen.kernel_id)
    assert decision.kernel_id == chosen.kernel_id and decision.stack_id == "triton"
    assert len(decision.reports) == len(decision.stack_reports) == 1
    assert "其他栈未评估" in ";".join(decision.stack_reports[0].reasons)
    text = explain_dispatch(SIG, HW, registry=registry, kernel_id=chosen.kernel_id)
    assert "其他配置未评估" in text and "第一级" in text and "第二级" in text


def test_explicit_ineligible_config_does_not_fall_back_to_another_config():
    chosen = descriptor("triton/bad", capability=CapabilitySpec(SIG.op, dtypes=("int8",)))
    registry = registry_of(chosen, descriptor("triton/ok"))
    with pytest.raises(DispatchError, match="显式指定 kernel") as caught:
        dispatch(SIG, HW, registry=registry, kernel_id=chosen.kernel_id)
    assert "第一级" in str(caught.value) and "第二级" in str(caught.value)
    assert "dtype float32" in str(caught.value)
    assert "其他配置未评估" in str(caught.value)


def test_existing_tuning_champion_is_explained_at_both_levels(monkeypatch):
    a = descriptor("triton/a")
    b = descriptor("triton/b")
    registry = registry_of(a, b, descriptor("cann/a", Backend.CANN))
    monkeypatch.setattr(DISPATCH, "query_tuning_record", lambda *args, **kwargs: b.kernel_id)
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.stack_id == "triton" and decision.kernel_id == b.kernel_id
    assert decision.fallback_chain == ("cann/a", a.kernel_id)
    triton = stack_report(decision, "triton")
    assert triton.representative_kernel_id == b.kernel_id
    assert "tuning 冠军锁定此栈" in ";".join(triton.reasons)
    assert "栈内让位" in ";".join(triton.implementations[0].reasons)
    assert "栈内实测 tuning 记录命中" in ";".join(triton.implementations[1].reasons)
    assert "tuning 冠军所在栈" in ";".join(stack_report(decision, "cann").reasons)


def test_tuning_cannot_resurrect_ineligible_config_in_an_available_stack(monkeypatch):
    bad = descriptor("triton/a", capability=CapabilitySpec(SIG.op, dtypes=("int8",)))
    good = descriptor("triton/b")
    monkeypatch.setattr(DISPATCH, "query_tuning_record", lambda *args, **kwargs: bad.kernel_id)
    decision = dispatch(SIG, HW, registry=registry_of(bad, good))
    assert decision.kernel_id == good.kernel_id
    stack, = decision.stack_reports
    assert not stack.implementations[0].accepted and stack.implementations[1].accepted
    assert "不在当前可用候选中" in decision.reason


@pytest.mark.parametrize("hidden,expected", [(1, 128), (127, 128), (128, 128), (129, 256), (256, 256)])
def test_real_example_configs_select_by_actual_tile_capacity(hidden, expected):
    from examples.two_level_gemma import OP, configuration_registry

    production_before = tuple((d.kernel_id, d.fingerprint) for d in get_global_registry().descriptors())
    assert len(production_before) == DEFAULT_DESCRIPTOR_COUNT
    registry = configuration_registry(HW.soc_model)
    signature = replace(SIG, op=OP, shapes=((4, hidden),))
    decision = dispatch(signature, HW, registry=registry)
    assert decision.stack_id == "triton"
    assert decision.kernel_id == f"triton/t19a/gemma/block{expected}"
    assert len(stack_report(decision, "triton").implementations) == 2
    if hidden <= 128:
        assert decision.fallback_chain == ("triton/t19a/gemma/block256",)
        assert "kernel_id 字典序" in explain_dispatch(signature, HW, registry=registry)
    else:
        assert not stack_report(decision, "triton").implementations[0].accepted
        assert "hidden %d 不在支持区间 [1, 128]" % hidden in explain_dispatch(signature, HW, registry=registry)
    # The optional example must not add or mutate production registrations.
    assert tuple((d.kernel_id, d.fingerprint) for d in get_global_registry().descriptors()) == production_before
    assert not any(kernel_id.startswith("triton/t19a/") for kernel_id, _ in production_before)


@pytest.mark.parametrize("hidden", [0, 257])
def test_real_example_does_not_claim_support_outside_either_tile(hidden):
    from examples.two_level_gemma import OP, configuration_registry

    with pytest.raises(DispatchError, match="hidden"):
        dispatch(replace(SIG, op=OP, shapes=((4, hidden),)), HW,
                 registry=configuration_registry(HW.soc_model))


def test_stack_interfaces_are_in_the_reproducible_parity_ledger():
    from flashinfer.parity import ParityEntry, load_packaged_manifest
    from tools.update_parity_inventory import annotate

    class NoUpstreamExtension:
        def resolve(self, symbol):
            return None

    entries = {e.local: e for e in load_packaged_manifest("all").entries}
    for name in ("registry.ImplementationStack", "registry.KernelRegistry.stacks",
                 "registry.KernelRegistry.stack_id_for", "dispatch.StackReport"):
        entry = entries["flashinfer.runtime." + name]
        assert entry.api_kind == "npu_extension"
        assert ParityEntry.from_dict(annotate(entry.to_dict(), NoUpstreamExtension())) == entry
