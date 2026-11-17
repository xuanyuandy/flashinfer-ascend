"""attention accuracy reports regression coverage."""


import math
import unittest
from dataclasses import replace

from flashinfer.attention import (
    AttentionAccuracyBudget,
    AttentionAccuracyCorpus,
    AttentionAccuracyExpectationError,
    AttentionAccuracyReport,
    AttentionErrorTolerance,
    AttentionMode,
    AttentionPlanSpec,
    AttentionTrace,
    ReferenceAttentionResult,
    ReferenceKVData,
    ReferenceQuantizedKVData,
    ReferenceQuantizedTensor,
    ReferenceTensor,
    RaggedKVCacheSpec,
    SingleAttentionMetadata,
    compare_attention_results,
    build_attention_accuracy_corpus,
    evaluate_attention_accuracy,
)
from flashinfer.runtime import QuantSpec, SchemaError


def tensor(value, dtype="float32", device="cpu"):
    return ReferenceTensor.from_nested(value, dtype=dtype, device=device)


def paired_traces(dense_values, storage_values, *, scale=1.0):
    quant = QuantSpec(
        scheme="symmetric",
        storage_dtype="int8",
        compute_dtype="float32",
        accumulator_dtype="float32",
    )
    common = dict(
        mode=AttentionMode.SINGLE_PREFILL,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=1,
        head_dim_vo=1,
        q_dtype="float32",
        o_dtype="float32",
        sm_scale=1.0,
    )
    dense_spec = AttentionPlanSpec(kv_dtype="float32", **common)
    quantized_spec = AttentionPlanSpec(
        kv_dtype="int8", kv_quant_spec=quant, **common
    )
    metadata = SingleAttentionMetadata(1, 2)
    q = tensor([[[0.0]]])
    dense_cache = RaggedKVCacheSpec(2, 1, 1, 1, "float32", device="cpu")
    quantized_cache = RaggedKVCacheSpec(
        2, 1, 1, 1, "int8", device="cpu", quant_spec=quant
    )
    dense = AttentionTrace.capture(
        spec=dense_spec,
        metadata=metadata,
        q=q,
        kv_data=ReferenceKVData(
            dense_cache,
            (
                tensor([[[0.0]], [[0.0]]]),
                tensor([[[dense_values[0]]], [[dense_values[1]]]]),
            ),
        ),
    )
    quantized = AttentionTrace.capture(
        spec=quantized_spec,
        metadata=metadata,
        q=q,
        kv_data=ReferenceQuantizedKVData(
            quantized_cache,
            ReferenceQuantizedTensor(
                (2, 1, 1), tensor([[[0]], [[0]]], "int8"), tensor(scale), quant
            ),
            ReferenceQuantizedTensor(
                (2, 1, 1),
                tensor([[[storage_values[0]]], [[storage_values[1]]]], "int8"),
                tensor(scale),
                quant,
            ),
        ),
    )
    return dense, quantized


class AttentionAccuracyBudgetTests(unittest.TestCase):
    def test_builtin_accuracy_corpus_round_trips_and_replays_declared_verdicts(self):
        corpus = build_attention_accuracy_corpus()
        self.assertEqual(corpus.name, "attention-quantization-accuracy-v1")
        self.assertEqual(len(corpus.cases), 4)
        restored = AttentionAccuracyCorpus.from_json(corpus.to_json(indent=2))
        self.assertEqual(restored.to_json(), corpus.to_json())
        self.assertEqual(restored.fingerprint, corpus.fingerprint)
        reports = dict(restored.replay_all())
        self.assertTrue(reports["exact_int8"].quantization_within_budget)
        self.assertTrue(
            reports["lossy_asymmetric_uint8"].quantization_within_budget
        )
        self.assertTrue(
            reports["lossy_packed_int4_odd_dimension"].quantization_within_budget
        )
        self.assertFalse(
            reports["int8_scale_overflow_rejected"].quantization_within_budget
        )
        tampered = restored.cases[0]
        with self.assertRaises(AttentionAccuracyExpectationError):
            replace(tampered, expect_quantization_pass=False).evaluate()

    def test_budget_round_trip_and_fingerprint_are_stable(self):
        budget = AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(0.25, 0.01),
            quantization_lse=AttentionErrorTolerance(0.1, 0.02),
            backend_output=AttentionErrorTolerance(1e-4, 1e-3),
            backend_lse=AttentionErrorTolerance(2e-4, 2e-3),
        )
        restored = AttentionAccuracyBudget.from_dict(budget.to_dict())
        self.assertEqual(restored, budget)
        self.assertEqual(restored.fingerprint, budget.fingerprint)
        with self.assertRaisesRegex(SchemaError, "finite and non-negative"):
            AttentionErrorTolerance(-1.0, 0.0)
        invalid = budget.to_dict()
        invalid["schema_version"] = 2
        with self.assertRaisesRegex(SchemaError, "budget version"):
            AttentionAccuracyBudget.from_dict(invalid)

    def test_exact_int8_has_zero_quantization_and_backend_error(self):
        dense, quantized = paired_traces((1.0, 3.0), (1, 3))
        report = evaluate_attention_accuracy(
            dense, quantized, budget=AttentionAccuracyBudget()
        )
        self.assertTrue(report.passes)
        self.assertEqual(report.quantization.output.max_abs_error, 0.0)
        self.assertEqual(report.backend.output.max_abs_error, 0.0)
        self.assertEqual(len(report.fingerprint), 64)
        restored = AttentionAccuracyReport.from_dict(report.to_dict())
        self.assertEqual(restored, report)
        self.assertEqual(restored.fingerprint, report.fingerprint)
        tampered = report.to_dict()
        tampered["passes"] = False
        with self.assertRaisesRegex(SchemaError, "passes is inconsistent"):
            AttentionAccuracyReport.from_dict(tampered)
        malformed = report.to_dict()
        malformed["candidate_result_fingerprint"] = 7
        with self.assertRaisesRegex(SchemaError, "SHA-256"):
            AttentionAccuracyReport.from_dict(malformed)

    def test_lossy_quantization_uses_its_own_budget(self):
        dense, quantized = paired_traces((1.1, 2.1), (1, 2))
        budget = AttentionAccuracyBudget(
            quantization_output=AttentionErrorTolerance(atol=0.11),
        )
        report = evaluate_attention_accuracy(dense, quantized, budget=budget)
        self.assertTrue(report.quantization_within_budget)
        self.assertTrue(report.backend_within_budget)
        self.assertTrue(report.passes)
        self.assertAlmostEqual(report.quantization.output.max_abs_error, 0.1)

    def test_backend_error_does_not_get_charged_to_quantization(self):
        dense, quantized = paired_traces((1.1, 2.1), (1, 2))
        reference = quantized.replay()
        candidate = ReferenceAttentionResult(
            ReferenceTensor(
                reference.output.shape,
                (reference.output.data[0] + 0.2,),
                reference.output.dtype,
                "npu:0",
            ),
            reference.lse,
        )
        report = evaluate_attention_accuracy(
            dense,
            quantized,
            budget=AttentionAccuracyBudget(
                quantization_output=AttentionErrorTolerance(atol=0.11),
                backend_output=AttentionErrorTolerance(atol=0.01),
            ),
            candidate=candidate,
        )
        self.assertTrue(report.quantization_within_budget)
        self.assertFalse(report.backend_within_budget)
        self.assertFalse(report.passes)
        self.assertEqual(report.backend.output.tolerance_violations, 1)

    def test_scale_overflow_is_quantization_failure_not_backend_failure(self):
        dense, quantized = paired_traces(
            (1e308, 1e308), (127, 127), scale=1.5e306
        )
        report = evaluate_attention_accuracy(
            dense,
            quantized,
            budget=AttentionAccuracyBudget(
                quantization_output=AttentionErrorTolerance(atol=1e308)
            ),
        )
        self.assertFalse(report.quantization_within_budget)
        self.assertTrue(report.backend_within_budget)
        self.assertEqual(report.quantization.output.nonfinite_mismatches, 1)

    def test_nonfinite_semantics_and_trace_pair_identity_are_strict(self):
        expected = ReferenceAttentionResult(
            ReferenceTensor((3,), (math.nan, math.inf, -math.inf)), None
        )
        matching = ReferenceAttentionResult(
            ReferenceTensor(
                (3,), (math.nan, math.inf, -math.inf), device="npu:0"
            ),
            None,
        )
        metrics = compare_attention_results(expected, matching)
        self.assertTrue(metrics.within_budget)
        self.assertEqual(metrics.output.nonfinite_matches, 3)
        mismatching = ReferenceAttentionResult(
            ReferenceTensor((3,), (0.0, math.inf, math.inf)), None
        )
        metrics = compare_attention_results(expected, mismatching)
        self.assertFalse(metrics.within_budget)
        self.assertEqual(metrics.output.nonfinite_mismatches, 2)

        dense, quantized = paired_traces((1.0, 3.0), (1, 3))
        incompatible = replace(
            quantized, spec=replace(quantized.spec, window_left=0)
        )
        with self.assertRaisesRegex(SchemaError, "plan semantics"):
            evaluate_attention_accuracy(
                dense, incompatible, budget=AttentionAccuracyBudget()
            )


from flashinfer.attention import (
    AttentionAccuracyBudget,
    AttentionAccuracyCase,
    AttentionAccuracyCorpus,
    AttentionAccuracyBindingError,
    AttentionAccuracyDispatchBinding,
    AttentionTrace,
    ReferenceAttentionResult,
    ReferenceKVData,
    ReferenceTensor,
    bind_attention_accuracy_dispatch,
    build_attention_accuracy_corpus,
    evaluate_attention_accuracy,
    select_attention_dispatch,
)
from flashinfer.runtime import SchemaError
from tests.framework.test_attention_capability import bound_kernel, functional_profile, group_case, group_plan


def accuracy_dispatch_fixture():
    conformance_corpus, quantized_case = group_case()
    quantized_trace = quantized_case.trace
    quantized_kv = quantized_trace.kv_data
    dense_spec = replace(
        quantized_trace.spec, kv_dtype="float32", kv_quant_spec=None
    )
    dense_cache = replace(
        quantized_kv.spec, dtype="float32", quant_spec=None
    )
    dense_trace = AttentionTrace.capture(
        spec=dense_spec,
        metadata=quantized_trace.metadata,
        q=quantized_trace.q,
        kv_data=ReferenceKVData(
            dense_cache,
            (
                quantized_kv.key_data.dequantize(),
                quantized_kv.value_data.dequantize(),
            ),
        ),
        custom_mask_data=quantized_trace.custom_mask_data,
        alibi_slopes=quantized_trace.alibi_slopes,
        q_scale=quantized_trace.q_scale,
        k_scale=quantized_trace.k_scale,
        v_scale=quantized_trace.v_scale,
        logits_soft_cap=quantized_trace.logits_soft_cap,
        return_lse=quantized_trace.return_lse,
    )
    accuracy_case = AttentionAccuracyCase(
        "paged_decode_int8_group_exact_dequant",
        dense_trace,
        quantized_trace,
        AttentionAccuracyBudget(),
        True,
        "Synthetic Host binding fixture; not measured NPU evidence.",
    )
    accuracy_corpus = AttentionAccuracyCorpus(
        "synthetic-accuracy-dispatch-v1", (accuracy_case,)
    )
    candidate = quantized_trace.replay()
    report = evaluate_attention_accuracy(
        dense_trace,
        quantized_trace,
        budget=accuracy_case.budget,
        candidate=candidate,
    )
    profile = functional_profile()
    descriptor = bound_kernel(profile)
    receipt = select_attention_dispatch(
        group_plan(),
        (profile,),
        (descriptor,),
        profile.environment,
    )
    return (
        conformance_corpus,
        accuracy_case,
        accuracy_corpus,
        candidate,
        report,
        profile,
        descriptor,
        receipt,
    )


class AttentionAccuracyDispatchBindingTests(unittest.TestCase):
    def test_binding_round_trips_and_revalidates_the_full_authority_chain(self):
        (
            conformance_corpus,
            case,
            corpus,
            candidate,
            report,
            profile,
            descriptor,
            receipt,
        ) = accuracy_dispatch_fixture()
        binding = bind_attention_accuracy_dispatch(
            binding_id="synthetic-host-binding-v1",
            runner="host-contract-test",
            accuracy_case=case,
            accuracy_corpus=corpus,
            accuracy_report=report,
            candidate=candidate,
            dispatch_receipt=receipt,
            profile=profile,
            descriptor=descriptor,
            environment=profile.environment,
            conformance_corpus=conformance_corpus,
        )
        restored = AttentionAccuracyDispatchBinding.from_dict(binding.to_dict())
        self.assertEqual(restored, binding)
        self.assertEqual(restored.fingerprint, binding.fingerprint)
        restored.validate(
            case,
            corpus,
            report,
            candidate,
            receipt,
            profile,
            descriptor,
            profile.environment,
            conformance_corpus=conformance_corpus,
        )
        self.assertEqual(binding.kernel_id, descriptor.kernel_id)
        self.assertEqual(binding.artifact_fingerprint, receipt.artifact_fingerprint)
        self.assertEqual(binding.accuracy_report_fingerprint, report.fingerprint)
        with self.assertRaisesRegex(SchemaError, "backend is invalid"):
            replace(binding, backend="cuda-guess")

    def test_report_candidate_and_binding_tampering_are_rejected(self):
        (
            conformance_corpus,
            case,
            corpus,
            candidate,
            report,
            profile,
            descriptor,
            receipt,
        ) = accuracy_dispatch_fixture()
        changed_candidate = ReferenceAttentionResult(
            ReferenceTensor(
                candidate.output.shape,
                tuple(value + 0.25 for value in candidate.output.data),
                candidate.output.dtype,
                candidate.output.device,
            ),
            candidate.lse,
        )
        with self.assertRaisesRegex(
            AttentionAccuracyBindingError, "does not match replayed"
        ):
            bind_attention_accuracy_dispatch(
                binding_id="synthetic-host-binding-v1",
                runner="host-contract-test",
                accuracy_case=case,
                accuracy_corpus=corpus,
                accuracy_report=report,
                candidate=changed_candidate,
                dispatch_receipt=receipt,
                profile=profile,
                descriptor=descriptor,
                environment=profile.environment,
                conformance_corpus=conformance_corpus,
            )

        binding = bind_attention_accuracy_dispatch(
            binding_id="synthetic-host-binding-v1",
            runner="host-contract-test",
            accuracy_case=case,
            accuracy_corpus=corpus,
            accuracy_report=report,
            candidate=candidate,
            dispatch_receipt=receipt,
            profile=profile,
            descriptor=descriptor,
            environment=profile.environment,
            conformance_corpus=conformance_corpus,
        )
        stale = replace(binding, artifact_fingerprint="0" * 64)
        with self.assertRaisesRegex(
            AttentionAccuracyBindingError, "stale or inconsistent"
        ):
            stale.validate(
                case,
                corpus,
                report,
                candidate,
                receipt,
                profile,
                descriptor,
                profile.environment,
                conformance_corpus=conformance_corpus,
            )

    def test_cross_workload_and_expected_rejection_cases_cannot_bind(self):
        (
            conformance_corpus,
            _case,
            _corpus,
            _candidate,
            _report,
            profile,
            descriptor,
            receipt,
        ) = accuracy_dispatch_fixture()
        builtin = build_attention_accuracy_corpus()
        for case_id, reason in (
            ("exact_int8", "dispatch receipt revalidation failed"),
            ("int8_scale_overflow_rejected", "cannot bind runnable dispatch"),
        ):
            case = next(item for item in builtin.cases if item.case_id == case_id)
            candidate = case.quantized_trace.replay()
            report = evaluate_attention_accuracy(
                case.dense_trace,
                case.quantized_trace,
                budget=case.budget,
                candidate=candidate,
            )
            with self.subTest(case_id=case_id):
                with self.assertRaisesRegex(AttentionAccuracyBindingError, reason):
                    bind_attention_accuracy_dispatch(
                        binding_id="invalid-cross-workload",
                        runner="host-contract-test",
                        accuracy_case=case,
                        accuracy_corpus=builtin,
                        accuracy_report=report,
                        candidate=candidate,
                        dispatch_receipt=receipt,
                        profile=profile,
                        descriptor=descriptor,
                        environment=profile.environment,
                        conformance_corpus=conformance_corpus,
                    )

    def test_binding_schema_rejects_malformed_hashes(self):
        values = {name: "x" for name in AttentionAccuracyDispatchBinding.__dataclass_fields__}
        values["schema_version"] = 1
        with self.assertRaisesRegex(SchemaError, "SHA-256"):
            AttentionAccuracyDispatchBinding(**values)


from flashinfer.attention import (
    AttentionAccuracyExecutionBinding,
    AttentionAccuracyExecutionBindingError,
    AttentionHostBufferRegistry,
    AttentionLaunchSession,
    AttentionLeaseRegistry,
    ReferenceAttentionResult,
    ReferenceTensor,
    bind_attention_accuracy_dispatch,
    bind_attention_accuracy_execution,
    capture_attention_protocol,
)
from tests.framework.test_attention_launcher_provider import FakeProvider, registered_launcher, resolution_fixture


def execution_binding_fixture(completion_code=0):
    (
        conformance_corpus,
        case,
        accuracy_corpus,
        candidate,
        report,
        profile,
        descriptor,
        receipt,
    ) = accuracy_dispatch_fixture()
    dispatch_binding = bind_attention_accuracy_dispatch(
        binding_id="synthetic-host-binding-v1",
        runner="host-contract-test",
        accuracy_case=case,
        accuracy_corpus=accuracy_corpus,
        accuracy_report=report,
        candidate=candidate,
        dispatch_receipt=receipt,
        profile=profile,
        descriptor=descriptor,
        environment=profile.environment,
        conformance_corpus=conformance_corpus,
    )
    packet, _kernel, probe, load_evidence, resolved = resolution_fixture()
    provider = FakeProvider(probe, completion_code=completion_code)
    with capture_attention_protocol("accuracy-execution") as capture:
        session = AttentionLaunchSession(
            packet,
            resolved,
            AttentionLeaseRegistry(),
            AttentionHostBufferRegistry(),
            registered_launcher(resolved, load_evidence),
        )
        submit = session.submit(provider)
        completion = session.poll_completion(provider)
        session.release()
    return (
        dispatch_binding,
        candidate,
        packet,
        resolved,
        submit,
        completion,
        capture.traces[0],
    )


class AttentionAccuracyExecutionBindingTests(unittest.TestCase):
    def test_successful_provider_lifecycle_round_trips_and_revalidates(self):
        values = execution_binding_fixture()
        binding = bind_attention_accuracy_execution(
            binding_id="synthetic-execution-binding-v1",
            runner="host-contract-provider",
            accuracy_dispatch_binding=values[0],
            candidate=values[1],
            launch_packet=values[2],
            resolved_launcher=values[3],
            submit_result=values[4],
            completion_result=values[5],
            protocol_trace=values[6],
        )
        restored = AttentionAccuracyExecutionBinding.from_dict(binding.to_dict())
        self.assertEqual(restored, binding)
        self.assertEqual(restored.fingerprint, binding.fingerprint)
        restored.validate(*values)
        self.assertEqual(binding.launch_packet_fingerprint, values[2].fingerprint)
        self.assertEqual(binding.protocol_trace_fingerprint, values[6].fingerprint)
        self.assertEqual(binding.result_origin, "runner_declared_post_completion")

    def test_async_failure_cannot_bind_accuracy_result(self):
        values = execution_binding_fixture(completion_code=8)
        with self.assertRaisesRegex(
            AttentionAccuracyExecutionBindingError, "successful provider completion"
        ):
            bind_attention_accuracy_execution(
                binding_id="failed-execution",
                runner="host-contract-provider",
                accuracy_dispatch_binding=values[0],
                candidate=values[1],
                launch_packet=values[2],
                resolved_launcher=values[3],
                submit_result=values[4],
                completion_result=values[5],
                protocol_trace=values[6],
            )

    def test_candidate_provider_and_protocol_identity_drift_are_rejected(self):
        values = execution_binding_fixture()
        changed_candidate = ReferenceAttentionResult(
            ReferenceTensor(
                values[1].output.shape,
                tuple(value + 1.0 for value in values[1].output.data),
                values[1].output.dtype,
                values[1].output.device,
            ),
            values[1].lse,
        )
        variants = (
            (
                dict(candidate=changed_candidate),
                "candidate result does not match",
            ),
            (
                dict(
                    completion_result=replace(
                        values[5], launch_packet_fingerprint="f" * 64
                    )
                ),
                "identity chain is inconsistent",
            ),
            (
                dict(
                    protocol_trace=replace(
                        values[6], subject_fingerprint="0" * 64
                    )
                ),
                "subject or stream",
            ),
        )
        names = (
            "accuracy_dispatch_binding",
            "candidate",
            "launch_packet",
            "resolved_launcher",
            "submit_result",
            "completion_result",
            "protocol_trace",
        )
        base = dict(zip(names, values))
        for changes, reason in variants:
            current = dict(base)
            current.update(changes)
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(
                    AttentionAccuracyExecutionBindingError, reason
                ):
                    bind_attention_accuracy_execution(
                        binding_id="drifted-execution",
                        runner="host-contract-provider",
                        **current,
                    )

    def test_stored_binding_tamper_and_schema_errors_are_rejected(self):
        values = execution_binding_fixture()
        binding = bind_attention_accuracy_execution(
            binding_id="synthetic-execution-binding-v1",
            runner="host-contract-provider",
            accuracy_dispatch_binding=values[0],
            candidate=values[1],
            launch_packet=values[2],
            resolved_launcher=values[3],
            submit_result=values[4],
            completion_result=values[5],
            protocol_trace=values[6],
        )
        stale = replace(binding, protocol_trace_fingerprint="1" * 64)
        with self.assertRaisesRegex(
            AttentionAccuracyExecutionBindingError, "stale or inconsistent"
        ):
            stale.validate(*values)
        with self.assertRaisesRegex(SchemaError, "result_origin"):
            replace(binding, result_origin="attested")
        with self.assertRaisesRegex(SchemaError, "runtime_generation"):
            replace(binding, runtime_generation=0)
