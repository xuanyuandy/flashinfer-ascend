"""Semantic routing with synthetic host descriptors, not device support evidence."""

from dataclasses import replace
import importlib
import json

import pytest

from flashinfer.runtime import (
    Backend, CapabilityCase, CapabilitySpec, DeviceCapability, DispatchError,
    FeatureConstraint, HardwareSignature, InferenceSpec, InputSignature, KernelDescriptor,
    KernelRegistry, QuantSpec, SchemaError, ShapeRange, WorkloadSpec,
    build_signature, dispatch, explain_dispatch, functional_feature_catalog,
)
from flashinfer.runtime.manifest import descriptor_from_dict


OP = "test.inference.matmul"
HW = HardwareSignature("Ascend950", 32, 262144)
DTYPES = ("bfloat16",) * 3


@pytest.fixture(autouse=True)
def policies(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")


def signature(*, inference=None, dtypes=DTYPES, m=32, **kwargs):
    return InputSignature(OP, dtypes=dtypes, shapes=((m, 32), (32, 64), (m, 64)),
                          inference=inference, **kwargs)


def descriptor(name, *, case=None, backend=Backend.CANN, **capability):
    return KernelDescriptor(f"{backend.value}/inference/{name}", backend=backend,
                            capability=CapabilitySpec(OP, cases=() if case is None else (case,),
                                                      **capability))


def quant(**changes):
    return replace(QuantSpec("symmetric", "int8", "int8", "int32", granularity="token"), **changes)


def test_complete_dtype_signatures_do_not_form_a_cartesian_product():
    capability = CapabilitySpec(OP, dtype_signatures=(DTYPES, ("float16",) * 3))
    assert not capability.unsupported_reasons(signature(), HW)
    assert capability.unsupported_reasons(signature(dtypes=("bfloat16", "float16", "bfloat16")), HW)
    assert capability.unsupported_reasons(signature(dtypes=("bfloat16", "bfloat16")), HW)
    assert capability.unsupported_reasons(signature(dtypes=("",) * 3), HW)


@pytest.mark.parametrize("dtype,m,accepted", [
    ("bfloat16", 32, True), ("float16", 96, True),
    ("bfloat16", 96, False), ("float16", 32, False), ("float16", 80, False),
])
def test_conditional_shape_dtype_and_alignment_are_coherent(dtype, m, accepted):
    capability = CapabilitySpec(OP, cases=(
        CapabilityCase("small-bf16", dtype_signatures=(DTYPES,),
                       shape_ranges=(ShapeRange("M", 0, 0, 16, 64),)),
        CapabilityCase("large-fp16", dtype_signatures=(("float16",) * 3,),
                       shape_ranges=(ShapeRange("M", 0, 0, 64, 128),),
                       shape_multiples=((0, 0, 32),)),
    ))
    assert bool(capability.unsupported_reasons(signature(dtypes=(dtype,) * 3, m=m), HW)) != accepted


@pytest.mark.parametrize("field,value,allowed", [
    ("compute_dtype", "float16", "compute_dtypes"),
    ("accumulator_dtype", "float32", "accumulator_dtypes"),
    ("attention_mode", "decode", "attention_modes"),
    ("attention_variant", "gqa", "attention_variants"),
    ("kv_cache_type", "paged", "kv_cache_types"),
    ("mask_type", "packed_custom", "mask_types"),
    ("pos_encoding", "ROPE_LLAMA", "pos_encodings"),
    ("causal", False, "causal_values"),
    ("sliding_window", True, "sliding_window_values"),
    ("sparse_format", "block_sparse", "sparse_formats"),
    ("routing_method", "grouped_topk", "routing_methods"),
    ("moe_top_k", 8, "moe_top_k_values"),
    ("expert_parallel_size", 2, "expert_parallel_sizes"),
])
def test_semantic_features_require_explicit_declarations_and_metadata(field, value, allowed):
    request = signature(inference=InferenceSpec(**{field: value}))
    capability = CapabilitySpec(OP, cases=(CapabilityCase("supported", **{allowed: (value,)}),))
    assert not capability.unsupported_reasons(request, HW)
    assert capability.unsupported_reasons(signature(), HW), "unknown is not supported"
    assert CapabilitySpec(OP, cases=(CapabilityCase("undeclared"),)).unsupported_reasons(request, HW)
    assert CapabilitySpec(OP).unsupported_reasons(request, HW), "legacy records must not gain inferred features"


@pytest.mark.parametrize("field,enabled", [("logits_soft_cap", 50.0), ("deterministic", True)])
def test_execution_flags_require_verified_support(field, enabled):
    support = "supports_logits_soft_cap" if field == "logits_soft_cap" else "supports_deterministic"
    request = signature(inference=InferenceSpec(**{field: enabled}))
    assert CapabilitySpec(OP, cases=(CapabilityCase("no"),)).unsupported_reasons(request, HW)
    assert not CapabilitySpec(OP, cases=(CapabilityCase("yes", **{support: True}),)).unsupported_reasons(request, HW)


@pytest.mark.parametrize("field,requested,wrong", [
    ("fusion_ops", ("bias", "silu"), ("silu", "bias")),
    ("auxiliary_outputs", ("lse",), ()),
])
def test_ordered_fusion_and_output_contracts_are_exact(field, requested, wrong):
    capability = CapabilitySpec(OP, cases=(CapabilityCase("exact", **{field: requested}),))
    assert not capability.unsupported_reasons(signature(inference=InferenceSpec(**{field: requested})), HW)
    assert capability.unsupported_reasons(signature(inference=InferenceSpec(**{field: wrong})), HW)


@pytest.mark.parametrize("change", [
    dict(scale_dtype="float16"), dict(granularity="channel", axis=(1,)),
    dict(granularity="group", group_size=(64,), axis=(1,)),
    dict(scheme="asymmetric", has_zero_point=True), dict(physical_layout="packed_scales"),
    dict(compute_dtype="float16"), dict(accumulator_dtype="float32"),
])
def test_quantization_is_per_operand_and_matches_full_contract(change):
    qa, qb = quant(), quant(granularity="channel", axis=(0,))
    capability = CapabilitySpec(OP, cases=(CapabilityCase("a8w8", quantization=((0, qa), (1, qb))),))
    correct = signature(dtypes=("int8", "int8", "bfloat16"),
                        inference=InferenceSpec(quantization=((1, qb), (0, qa))))
    assert not capability.unsupported_reasons(correct, HW)
    for bindings in (((0, replace(qa, **change)), (1, qb)), ((1, qa), (0, qb)), ((0, qa),)):
        assert capability.unsupported_reasons(replace(correct, inference=InferenceSpec(quantization=bindings)), HW)
    assert capability.unsupported_reasons(signature(), HW)


def test_packed_quantization_order_is_part_of_route_identity():
    low = quant(storage_dtype="int4_packed", compute_dtype="float16", accumulator_dtype="float32",
                packing_order="low_nibble_first", granularity="group", group_size=(128,))
    high = replace(low, packing_order="high_nibble_first")
    registry = KernelRegistry((descriptor("low", case=CapabilityCase("low", quantization=((1, low),))),
                               descriptor("high", case=CapabilityCase("high", quantization=((1, high),)))))
    for spec, expected in ((low, "low"), (high, "high")):
        request = signature(dtypes=("float16", "uint8", "float16"),
                            inference=InferenceSpec(quantization=((1, spec),)))
        assert dispatch(request, HW, registry=registry).kernel_id.endswith(expected)


def test_missing_case_attributes_and_shapes_fail_closed():
    capability = CapabilitySpec(OP, cases=(CapabilityCase("transpose",
        specializations=(("transpose_b", ("True",)),), shape_multiples=((1, -1, 16),)),))
    assert capability.unsupported_reasons(signature(), HW)
    good = signature(specializations=(("transpose_b", "True"),))
    assert not capability.unsupported_reasons(good, HW)
    assert capability.unsupported_reasons(replace(good, shapes=()), HW)
    assert capability.unsupported_reasons(replace(good, specializations=(("transpose_b", "True"),
                                                                 ("transpose_b", "False"))), HW)


def test_auto_forced_backend_and_kernel_id_all_filter_before_tuning(monkeypatch):
    module = importlib.import_module("flashinfer.runtime.dispatch")
    good = descriptor("good", case=CapabilityCase("bf16", compute_dtypes=("bfloat16",)))
    bad = descriptor("bad", backend=Backend.TRITON,
                     case=CapabilityCase("fp16", compute_dtypes=("float16",)))
    registry = KernelRegistry((bad, good))
    request = signature(inference=InferenceSpec(compute_dtype="bfloat16"))
    monkeypatch.setattr(module, "query_tuning_record", lambda *a, **k: bad.kernel_id)
    assert dispatch(request, HW, registry=registry).kernel_id == good.kernel_id
    assert dispatch(request, HW, backend="cann", registry=registry).kernel_id == good.kernel_id
    for kwargs in (dict(backend="triton"), dict(kernel_id=bad.kernel_id)):
        with pytest.raises(DispatchError, match="compute_dtype"):
            dispatch(request, HW, registry=registry, **kwargs)
    decision = dispatch(request, HW, registry=registry)
    assert any(not report.accepted and "compute_dtype" in str(report.reasons) for report in decision.reports)
    trace = explain_dispatch(request, HW, registry=registry)
    assert bad.kernel_id in trace and "compute_dtype" in trace


def test_cache_and_serialization_include_semantics_without_changing_legacy_payloads():
    legacy = signature()
    assert replace(legacy, inference=InferenceSpec()) == legacy
    assert "inference" not in legacy.to_dict()
    registry = KernelRegistry(tuple(descriptor(dtype, case=CapabilityCase(dtype, compute_dtypes=(dtype,)))
                                    for dtype in ("float16", "bfloat16")))
    for dtype in ("float16", "bfloat16"):
        request = replace(legacy, inference=InferenceSpec(compute_dtype=dtype))
        assert InputSignature(**json.loads(json.dumps(request.to_dict()))) == request
        assert hash(InputSignature(**request.to_dict())) == hash(request)
        first = dispatch(request, HW, registry=registry)
        assert not first.cache_hit and first.kernel_id.endswith(dtype)
        assert dispatch(request, HW, registry=registry).cache_hit


def test_manifest_and_workload_path_keep_extended_capability_and_identity():
    request = InferenceSpec(compute_dtype="int8", quantization=((0, quant()),))
    kernel = descriptor("quant", case=CapabilityCase("int8", compute_dtypes=("int8",),
                                                     quantization=request.quantization))
    restored = descriptor_from_dict(json.loads(json.dumps(kernel.to_dict())))
    assert restored.capability == kernel.capability and restored.fingerprint == kernel.fingerprint
    assert replace(kernel, capability=CapabilitySpec(OP)).fingerprint != kernel.fingerprint
    workload = WorkloadSpec(OP, dtypes=("int8", "bfloat16", "bfloat16"), inference=request)
    assert workload.quant_specs == (quant(),)
    assert WorkloadSpec.from_dict(workload.to_dict()) == workload
    registry = KernelRegistry((kernel,))
    assert registry.select(workload, DeviceCapability("Ascend950")).kernel_id == kernel.kernel_id
    with pytest.raises(DispatchError):
        registry.select(replace(workload, inference=None), DeviceCapability("Ascend950"))


def test_workload_quantization_cannot_conflict_with_explicit_tensor_bindings():
    request = InferenceSpec(quantization=((0, quant()),))
    with pytest.raises(SchemaError, match="quantization contradicts"):
        WorkloadSpec(OP, quant_specs=(quant(scale_dtype="float16"),), inference=request)
    with pytest.raises(SchemaError, match="quantization contradicts"):
        WorkloadSpec(OP, quant_specs=(quant(),), inference=InferenceSpec(compute_dtype="float16"))


@pytest.mark.parametrize("structured", [False, True])
def test_tuning_roundtrip_is_feature_and_shape_specific(monkeypatch, tmp_path, structured):
    from flashinfer.tuning import (BenchmarkCandidate, BenchmarkConfig, collect_tuning_record,
                                  save_tuning_record, read_tuning_record, reload_tuning_records)
    bench = importlib.import_module("flashinfer.tuning.benchmark")
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", str(tmp_path / "synthetic.sqlite3"))
    kernels = tuple(descriptor(name, case=CapabilityCase("compute", compute_dtypes=("bfloat16",)),
                                shape_ranges=(ShapeRange("M", 0, 0, 16, 128),)) for name in ("a", "b"))
    registry = KernelRegistry(kernels)
    def inference(dtype):
        return (InferenceSpec(features=(("compute.dtype", dtype),)) if structured
                else InferenceSpec(compute_dtype=dtype))

    request = signature(inference=inference("bfloat16"))
    state = {"calls": 0, "duration": 0}

    def prepare(name, iterations):
        state["duration"] = 100_000 if name.endswith("a") else 200_000
        return lambda: None

    def clock():
        state["calls"] += 1
        return 0 if state["calls"] % 2 else state["duration"]

    monkeypatch.setattr(bench.time, "perf_counter_ns", clock)
    monkeypatch.setattr(bench.time, "sleep", lambda _: None)
    record = collect_tuning_record(request, HW,
        [BenchmarkCandidate(k, lambda n, key=k.kernel_id: prepare(key, n)) for k in kernels],
        synchronize=lambda: None, config=BenchmarkConfig(warmup=2, rounds=25, iterations=1),
        metadata={"fixture": "SYNTHETIC ONLY; NOT PERFORMANCE EVIDENCE"})
    save_tuning_record(record)
    assert read_tuning_record(request, HW, registry=registry) is not None
    assert record["bucket"]["shape_ranges"] == [], "conditional capability must not widen a measured bucket"
    assert read_tuning_record(signature(m=64, inference=request.inference), HW, registry=registry) is None
    assert read_tuning_record(replace(request, inference=inference("float16")),
                              HW, registry=registry) is None
    reload_tuning_records()


@pytest.mark.parametrize("factory,kwargs", [
    (InferenceSpec, dict(quantization=((0, quant()), (0, quant())))),
    (InferenceSpec, dict(quantization=((True, quant()),))),
    (InferenceSpec, dict(fusion_ops="silu")), (InferenceSpec, dict(causal=1)),
    (InferenceSpec, dict(expert_parallel_size=True)), (InferenceSpec, dict(moe_top_k=0)),
    (InferenceSpec, dict(logits_soft_cap=float("nan"))),
    (CapabilityCase, dict(name="x", dtype_signatures=("float16",))),
    (CapabilityCase, dict(name="x", shape_multiples=((0, 0, 0),))),
    (CapabilityCase, dict(name="x", supports_deterministic=1)),
    (CapabilityCase, dict(name="x", causal_values=(1,))),
    (CapabilitySpec, dict(op=OP, cases=(CapabilityCase("x"), CapabilityCase("x")))),
])
def test_malformed_feature_contracts_are_rejected(factory, kwargs):
    with pytest.raises(SchemaError):
        factory(**kwargs)


@pytest.mark.parametrize("factory,payload", [
    (InferenceSpec, {"unknown": 1}), (CapabilityCase, {"name": "x", "unknown": 1}),
    (CapabilityCase, {"name": "x", "shape_ranges": None}),
    (CapabilitySpec, {"op": OP, "cases": [{"name": "x", "unknown": 1}]}),
])
def test_unknown_or_malformed_serialized_fields_are_rejected(factory, payload):
    with pytest.raises(SchemaError):
        factory.from_dict(payload)


def test_metadata_builder_and_workload_reject_contradictory_semantics():
    tensor = type("Metadata", (), {"dtype": "bfloat16", "shape": (32, 32)})()
    request = InferenceSpec(causal=True, pos_encoding="ALIBI")
    assert build_signature(OP, (tensor,), inference=request).inference == request
    with pytest.raises(ValueError, match="contradicts"):
        build_signature(OP, (tensor,), inference=request, specializations={"causal": False})
    with pytest.raises(SchemaError, match="contradicts"):
        WorkloadSpec(OP, inference=request)


@pytest.mark.parametrize("feature", functional_feature_catalog(), ids=lambda item: item.name)
def test_catalogued_dimensions_match_without_operator_specific_dispatch_code(feature):
    value = feature.choices[0] if feature.choices else {
        "string": "float32", "bool": False, "positive_int": 2,
    }[feature.value_type]
    facts = InferenceSpec(features=((feature.name, value),))
    capability = CapabilitySpec(OP, cases=(CapabilityCase("feature", features=(
        FeatureConstraint(feature.name, (value,)),)),))
    assert not capability.unsupported_reasons(signature(inference=facts), HW)
    assert capability.unsupported_reasons(signature(), HW)
    assert CapabilitySpec(OP, cases=(CapabilityCase("undeclared"),)).unsupported_reasons(
        signature(inference=facts), HW)
    assert InferenceSpec.from_dict(json.loads(json.dumps(facts.to_dict()))) == facts
    assert CapabilitySpec.from_dict(json.loads(json.dumps(capability.to_dict()))) == capability


def test_functional_gemm_attention_scenarios_route_and_reject_missing_facts():
    from tools.routing_capability_demo import HARDWARE, functional_scenarios, run_functional

    scenarios = functional_scenarios()
    registry = KernelRegistry(kernel for _, _, kernel in scenarios)
    evidence = run_functional()
    assert evidence["status"] == "PASS" and evidence["kernel_launches"] == 0
    assert {request.op for _, request, _ in scenarios} == {
        "example.functional.gemm", "example.functional.attention"}
    assert all(not request.shapes for _, request, _ in scenarios), "functional matching needs no M/N/K"
    for name, request, kernel in scenarios:
        # Every requested dimension is required; another case cannot donate it.
        for key, _ in request.inference.features:
            partial = replace(request.inference, features=tuple(
                fact for fact in request.inference.features if fact[0] != key))
            with pytest.raises(DispatchError, match="unsupported or missing"):
                dispatch(replace(request, inference=partial), HARDWARE, registry=registry)
        for backend in (Backend.CANN, Backend.ASCENDC, Backend.TRITON):
            if backend != kernel.backend:
                with pytest.raises(DispatchError):
                    dispatch(request, HARDWARE, backend=backend, registry=registry)
        roundtrip = descriptor_from_dict(json.loads(json.dumps(kernel.to_dict())))
        assert roundtrip.capability == kernel.capability and roundtrip.fingerprint == kernel.fingerprint


def test_functional_cases_do_not_mix_stage_cache_and_mask_from_different_cases():
    capability = CapabilitySpec(OP, cases=tuple(
        CapabilityCase(name, features=tuple(FeatureConstraint(k, (v,)) for k, v in features))
        for name, features in (
            ("prefill-dense", (("attention.mode", "prefill"), ("kv_cache.kind", "dense"),
                               ("attention.causal", True))),
            ("decode-paged", (("attention.mode", "decode"), ("kv_cache.kind", "paged"),
                              ("attention.causal", False))),
        )))
    for mode, cache, causal in (("prefill", "dense", True), ("decode", "paged", False)):
        facts = InferenceSpec(features=(("attention.mode", mode), ("kv_cache.kind", cache),
                                        ("attention.causal", causal)))
        assert not capability.unsupported_reasons(signature(inference=facts), HW)
    wrong = InferenceSpec(features=(("attention.mode", "prefill"), ("kv_cache.kind", "paged"),
                                    ("attention.causal", False)))
    assert capability.unsupported_reasons(signature(inference=wrong), HW)


def test_named_features_use_the_same_matcher_as_flat_compatibility_fields():
    flat = CapabilitySpec(OP, cases=(CapabilityCase("bf16", compute_dtypes=("bfloat16",)),))
    typed = CapabilitySpec(OP, cases=(CapabilityCase("bf16", features=(
        FeatureConstraint("compute.dtype", ("bfloat16",)),)),))
    for facts in (InferenceSpec(compute_dtype="bfloat16"),
                  InferenceSpec(features=(("compute.dtype", "bfloat16"),))):
        assert not flat.unsupported_reasons(signature(inference=facts), HW)
        assert not typed.unsupported_reasons(signature(inference=facts), HW)
    with pytest.raises(SchemaError, match="contradicts"):
        InferenceSpec(compute_dtype="float16", features=(("compute.dtype", "bfloat16"),))
    with pytest.raises(SchemaError, match="contradicts"):
        CapabilityCase("conflict", compute_dtypes=("float16",), features=(
            FeatureConstraint("compute.dtype", ("bfloat16",)),))


def test_structured_features_participate_in_decision_cache_and_workload_identity():
    facts = InferenceSpec(features=(("gemm.mode", "dense"),))
    kernels = tuple(descriptor(mode, case=CapabilityCase(mode, features=(
        FeatureConstraint("gemm.mode", (mode,)),))) for mode in ("dense", "grouped"))
    registry = KernelRegistry(kernels)
    first = signature(inference=facts)
    second = replace(first, inference=InferenceSpec(features=(("gemm.mode", "grouped"),)))
    assert dispatch(first, HW, registry=registry).kernel_id.endswith("dense")
    assert not dispatch(second, HW, registry=registry).cache_hit
    assert dispatch(first, HW, registry=registry).cache_hit
    assert InputSignature(**json.loads(json.dumps(first.to_dict()))) == first
    assert first.to_dict() != second.to_dict()
    workload = WorkloadSpec(OP, inference=facts)
    assert KernelRegistry(kernels).select(workload, DeviceCapability("Ascend950")).kernel_id.endswith("dense")
    with pytest.raises(SchemaError, match="contradicts"):
        WorkloadSpec(OP, inference=InferenceSpec(features=(("attention.causal", True),)))
    with pytest.raises(ValueError, match="contradicts"):
        replace(first, inference=InferenceSpec(features=(("attention.causal", True),)),
                specializations=(("causal", "False"),))


def test_gemm_attention_scenarios_reject_wrong_quantization_fusion_and_outputs():
    from tools.routing_capability_demo import HARDWARE, functional_scenarios

    scenarios = functional_scenarios()
    registry = KernelRegistry(kernel for _, _, kernel in scenarios)
    for _, request, _ in scenarios:
        facts = request.inference
        for index, quant_spec in facts.quantization:
            for changed in (replace(quant_spec, scale_dtype="float16"),
                            replace(quant_spec, scheme="asymmetric", has_zero_point=True)):
                bindings = tuple((i, changed if i == index else spec) for i, spec in facts.quantization)
                with pytest.raises(DispatchError):
                    dispatch(replace(request, inference=replace(facts, quantization=bindings)),
                             HARDWARE, registry=registry)
            if quant_spec.packing_order:
                bindings = tuple((i, replace(spec, packing_order="high_nibble_first") if i == index else spec)
                                 for i, spec in facts.quantization)
                with pytest.raises(DispatchError):
                    dispatch(replace(request, inference=replace(facts, quantization=bindings)),
                             HARDWARE, registry=registry)
        if facts.fusion_ops:
            with pytest.raises(DispatchError):
                dispatch(replace(request, inference=replace(facts, fusion_ops=tuple(reversed(facts.fusion_ops)))),
                         HARDWARE, registry=registry)
        if facts.auxiliary_outputs:
            with pytest.raises(DispatchError):
                dispatch(replace(request, inference=replace(facts, auxiliary_outputs=())),
                         HARDWARE, registry=registry)


@pytest.mark.parametrize("factory,kwargs", [
    (InferenceSpec, dict(features=(("unknown.feature", True),))),
    (InferenceSpec, dict(features=(("attention.causal", 1),))),
    (InferenceSpec, dict(features=(("moe.top_k", True),))),
    (InferenceSpec, dict(features=(("moe.top_k", 0),))),
    (InferenceSpec, dict(features=(("kv_cache.kind", "page_typo"),))),
    (InferenceSpec, dict(features=(("attention.mode", "batch_decode_paged"),))),
    (InferenceSpec, dict(features=(("gemm.mode", "dense"), ("gemm.mode", "dense")))),
    (FeatureConstraint, dict(name="gemm.mode", allowed_values=())),
    (FeatureConstraint, dict(name="gemm.mode", allowed_values="dense")),
    (FeatureConstraint, dict(name="attention.causal", allowed_values=(0,))),
    (CapabilityCase, dict(name="x", features=(FeatureConstraint("gemm.mode", ("dense",)),) * 2)),
])
def test_unknown_duplicate_or_mistyped_functional_features_fail_closed(factory, kwargs):
    with pytest.raises(SchemaError):
        factory(**kwargs)
