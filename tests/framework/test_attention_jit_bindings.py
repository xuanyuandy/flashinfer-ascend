"""attention jit bindings regression coverage."""


import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import flashinfer
from flashinfer.attention import (
    AttentionOperatorIntegrationError,
    AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionStateError,
    BatchAttention,
)
from flashinfer.jit import (
    ConfiguredJitArtifactVerifier,
    JitCacheIndex,
    JitCompilationPolicy,
    verify_jit_cache_record_payload,
)
from flashinfer.attention.jit import (
    ConfiguredAttentionJitArtifactResolver,
    resolve_attention_jit_plan,
)
from flashinfer.runtime import ArtifactVerificationError, SchemaError
from tests.framework.test_public_batch_attention_provider_runtime import plan_public_wrapper
from tests.support.attention_provider import FakeLogicalRunAdapter, build_components, framework_plan
from tests.framework.test_attention_jit_framework import runtime_environment
from tests.framework.test_attention_jit_active_plan import FakeJitAuthorityResolver, FakeJitAutoResolver, RecordingJitPlanResolver, SyntheticArtifactReader, cache_payload, cache_record, jit_receipt, package_jit_implementation


def direct_artifact_case():
    plan = framework_plan()
    environment = runtime_environment()
    receipt = jit_receipt(plan, environment, "cann")
    # The plan resolver creates the exact module spec before cache lookup.  A
    # first recipe-only pass lets the synthetic fixture create matching bytes.
    empty = resolve_attention_jit_plan(
        plan,
        receipt,
        environment,
        JitCacheIndex(),
        JitCompilationPolicy.CACHE_ONLY,
    )
    record = cache_record(empty.module_spec.jit_spec)
    cache = JitCacheIndex((record,))
    binding = resolve_attention_jit_plan(
        plan,
        receipt,
        environment,
        cache,
        JitCompilationPolicy.CACHE_ONLY,
    )
    payload = cache_payload(binding.module_spec.jit_spec)
    reader = SyntheticArtifactReader({record.artifact.fingerprint: payload})
    resolver = ConfiguredAttentionJitArtifactResolver(
        cache,
        ConfiguredJitArtifactVerifier("checkpoint032.synthetic", reader),
    )
    return plan, binding, record, cache, payload, reader, resolver


class NonBytesReader:
    def read(self, artifact):
        return "not-bytes"


def batch_wrapper(resolver):
    registry = AttentionOperatorRuntimeResolverRegistry((("npu", resolver),))
    with patch(
        "flashinfer.attention.holistic._operator_runtime_resolvers",
        registry,
    ):
        return BatchAttention(kv_layout="NHD", device="npu:0")


class AttentionJitArtifactVerificationTests(unittest.TestCase):
    """Checkpoint 032: byte identity is required before runtime publication."""

    def test_public_api_has_no_artifact_or_loader_controls(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("artifact", parameters)
            self.assertNotIn("loader", parameters)
        self.assertFalse(hasattr(flashinfer.jit, "load_jit_module"))
        self.assertFalse(hasattr(flashinfer.jit, "compile_jit_spec"))

    def test_exact_payload_produces_record_bound_receipt(self):
        _, binding, record, _, _, reader, resolver = direct_artifact_case()
        artifact_binding = resolver.resolve(binding)
        artifact_binding.validate_plan_binding(binding)
        artifact_binding.verification.validate_record(record)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(
            artifact_binding.verification.payload_digest,
            record.artifact.digest,
        )

    def test_wrong_size_and_digest_fail_closed(self):
        _, _, record, _, payload, _, _ = direct_artifact_case()
        with self.assertRaisesRegex(ArtifactVerificationError, "size"):
            verify_jit_cache_record_payload(
                record,
                payload[:-1],
                verifier_id="checkpoint032.synthetic",
            )
        corrupted = payload[:-1] + bytes([payload[-1] ^ 1])
        with self.assertRaisesRegex(ArtifactVerificationError, "digest"):
            verify_jit_cache_record_payload(
                record,
                corrupted,
                verifier_id="checkpoint032.synthetic",
            )

    def test_reader_must_return_bytes(self):
        _, binding, _, cache, _, _, _ = direct_artifact_case()
        resolver = ConfiguredAttentionJitArtifactResolver(
            cache,
            ConfiguredJitArtifactVerifier("checkpoint032.nonbytes", NonBytesReader()),
        )
        with self.assertRaisesRegex(TypeError, "must return bytes"):
            resolver.resolve(binding)

    def test_cache_record_must_still_match_resolution(self):
        _, binding, _, _, _, _, _ = direct_artifact_case()
        reader = SyntheticArtifactReader({})
        resolver = ConfiguredAttentionJitArtifactResolver(
            JitCacheIndex(),
            ConfiguredJitArtifactVerifier("checkpoint032.empty", reader),
        )
        with self.assertRaisesRegex(SchemaError, "record disappeared"):
            resolver.resolve(binding)
        self.assertEqual(reader.calls, [])

    def test_missing_artifact_resolver_stops_before_plan_or_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(components["events"], environment)
        implementation = AttentionOperatorPackageRuntimeImplementation(
            priority=base.priority,
            package_resolver=base._package_resolver,
            plan_gate=components["gate"],
            authority_resolver=FakeJitAuthorityResolver(
                components["events"], environment
            ),
            logical_factory=components["factory"],
            logical_run_adapter=FakeLogicalRunAdapter(),
            tensor_materializer=components["materializer"],
            jit_plan_resolver=jit_resolver,
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT artifact resolver"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_corrupt_artifact_stops_before_package_import(self):
        components, implementation, _, artifact_resolver, module_resolver = (
            package_jit_implementation(artifact_mode="corrupt")
        )
        with self.assertRaises(ArtifactVerificationError):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(artifact_resolver.calls, 1)
        self.assertEqual(module_resolver.calls, 0)
        self.assertIn("artifact_verify", components["events"])
        self.assertNotIn("resolve_callable", components["events"])
        self.assertEqual(components["loader"].resolve_calls, 0)

    def test_public_plan_owns_artifact_binding_and_run_does_not_reread(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        artifact_binding = runtime.jit_artifact_binding
        self.assertEqual(
            runtime.operator_session.active_plan.jit_artifact_binding_fingerprint,
            artifact_binding.fingerprint,
        )
        self.assertEqual(resolver.artifact_events, ["artifact_verify"])
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertEqual(resolver.artifact_events, ["artifact_verify"])

    def test_corrupt_replan_preserves_previous_plan_artifact_and_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_artifact = runtime.jit_artifact_binding
        old_executor = resolver.executors[0]

        resolver.artifact_mode = "corrupt"
        with self.assertRaises(ArtifactVerificationError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_artifact_binding, old_artifact)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_artifact_binding_drift_before_executor(self):
        first_resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(first_resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = first_resolver.executors[0]

        other = batch_wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_artifact_binding = (
            other._operator_runtime.jit_artifact_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "artifact active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_artifact_identity_changes_active_plan_fingerprint(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        active = wrapper._operator_runtime.operator_session.active_plan
        changed = replace(
            active,
            jit_artifact_binding_fingerprint="0" * 64,
        )
        self.assertNotEqual(active.fingerprint, changed.fingerprint)


from flashinfer.attention.schema import AttentionMode
from flashinfer.jit import JitCacheIndex, JitLoadedModule
from flashinfer.attention.jit import (
    ConfiguredAttentionJitModuleResolver,
    attention_jit_entry_points,
)
from flashinfer.runtime import SchemaError
from tests.framework.test_attention_jit_active_plan import FakeJitAuthorityResolver, FakeJitAutoResolver, RecordingJitArtifactResolver, RecordingJitPlanResolver, SyntheticModuleLoader, package_jit_implementation


class LyingIdentityLoader(SyntheticModuleLoader):
    def load(self, record, verification, required_entry_points):
        loaded = super().load(record, verification, required_entry_points)
        receipt = replace(loaded.receipt, loader_id="different.loader")
        return JitLoadedModule(receipt, loaded.opaque_module)


class AttentionJitModuleLoadingTests(unittest.TestCase):
    """Checkpoint 033: exact loaded-module identity remains wrapper-owned."""

    def loaded_case(self, *, symbol_mode="exact", loader=None):
        plan, binding, record, cache, _, _, artifact_resolver = direct_artifact_case()
        artifact_binding = artifact_resolver.resolve(binding)
        if loader is None:
            loader = SyntheticModuleLoader(symbol_mode=symbol_mode)
        resolver = ConfiguredAttentionJitModuleResolver(cache, loader)
        return (
            plan,
            binding,
            artifact_binding,
            record,
            cache,
            loader,
            resolver,
        )

    def test_entry_point_contract_matches_flashinfer_module_shape(self):
        self.assertEqual(
            attention_jit_entry_points(AttentionMode.SINGLE_PREFILL),
            ("run",),
        )
        self.assertEqual(
            attention_jit_entry_points(AttentionMode.SINGLE_DECODE),
            ("run",),
        )
        for mode in (
            AttentionMode.BATCH_PREFILL_PAGED,
            AttentionMode.BATCH_PREFILL_RAGGED,
            AttentionMode.BATCH_DECODE_PAGED,
            AttentionMode.BATCH_MIXED_PAGED,
        ):
            self.assertEqual(attention_jit_entry_points(mode), ("plan", "run"))

    def test_exact_loader_receipt_binds_artifact_and_symbols(self):
        _, binding, artifact, record, _, loader, resolver = self.loaded_case()
        module_binding = resolver.resolve(binding, artifact)
        module_binding.validate_bindings(binding, artifact)
        module_binding.loaded_module.receipt.validate(
            record,
            artifact.verification,
            binding.module_spec.jit_spec.entry_points,
        )
        self.assertEqual(
            tuple(item.name for item in module_binding.loaded_module.receipt.symbols),
            ("plan", "run"),
        )
        self.assertEqual(len(loader.calls), 1)

    def test_missing_or_extra_symbol_fails_closed(self):
        for mode in ("missing", "extra"):
            _, binding, artifact, _, _, loader, resolver = self.loaded_case(
                symbol_mode=mode
            )
            with self.assertRaisesRegex(SchemaError, "artifact and symbols"):
                resolver.resolve(binding, artifact)
            self.assertEqual(len(loader.calls), 1)

    def test_loader_cannot_change_declared_identity(self):
        loader = LyingIdentityLoader()
        _, binding, artifact, _, _, _, resolver = self.loaded_case(loader=loader)
        with self.assertRaisesRegex(SchemaError, "declared identity"):
            resolver.resolve(binding, artifact)

    def test_cache_record_must_survive_until_load(self):
        _, binding, artifact, _, _, loader, _ = self.loaded_case()
        resolver = ConfiguredAttentionJitModuleResolver(JitCacheIndex(), loader)
        with self.assertRaisesRegex(SchemaError, "disappeared before load"):
            resolver.resolve(binding, artifact)
        self.assertEqual(loader.calls, [])

    def test_missing_module_resolver_stops_before_jit_and_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(components["events"], environment)
        artifact_resolver = RecordingJitArtifactResolver(
            components["events"], jit_resolver
        )
        implementation = AttentionOperatorPackageRuntimeImplementation(
            priority=base.priority,
            package_resolver=base._package_resolver,
            plan_gate=components["gate"],
            authority_resolver=FakeJitAuthorityResolver(
                components["events"], environment
            ),
            logical_factory=components["factory"],
            logical_run_adapter=FakeLogicalRunAdapter(),
            tensor_materializer=components["materializer"],
            jit_plan_resolver=jit_resolver,
            jit_artifact_resolver=artifact_resolver,
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT module resolver"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertEqual(artifact_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_module_load_precedes_package_callable_import(self):
        components, implementation, _, _, module_resolver = package_jit_implementation()
        resolved = implementation.resolve(framework_plan(), "npu:0")
        events = components["events"]
        self.assertIsNotNone(resolved.jit_module_binding)
        self.assertLess(events.index("artifact_verify"), events.index("module_load"))
        self.assertLess(events.index("module_load"), events.index("resolve_callable"))
        self.assertEqual(module_resolver.calls, 1)

    def test_public_plan_owns_module_and_run_does_not_reload(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        module_binding = runtime.jit_module_binding
        self.assertEqual(
            runtime.operator_session.active_plan.jit_module_binding_fingerprint,
            module_binding.fingerprint,
        )
        self.assertEqual(resolver.module_events, ["module_load"])
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertEqual(resolver.module_events, ["module_load"])

    def test_failed_module_replan_preserves_old_runtime(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_module = runtime.jit_module_binding
        old_executor = resolver.executors[0]

        resolver.module_symbol_mode = "missing"
        with self.assertRaises(SchemaError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_module_binding, old_module)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_loaded_module_drift_before_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = resolver.executors[0]

        other = batch_wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_module_binding = (
            other._operator_runtime.jit_module_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "module active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_public_surface_has_no_module_handle_or_build_control(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("module", parameters)
            self.assertNotIn("jit", parameters)
        # T18B adds an explicit compiler-integration entry; Attention plan/run
        # still have no user-controlled module handle or JIT switch.
        self.assertTrue(callable(flashinfer.jit.build_and_load))
        self.assertFalse(hasattr(flashinfer.jit, "default_module_loader"))


from flashinfer.runtime import Backend, SchemaError
from tests.framework.test_attention_jit_active_plan import FakeJitAuthorityResolver, FakeJitAutoResolver, RecordingJitArtifactResolver, RecordingJitModuleResolver, RecordingJitPlanResolver, package_jit_implementation


class AttentionJitExecutorBindingTests(unittest.TestCase):
    """Loaded JIT symbols require a private, active-plan-bound executor."""

    def test_exact_binding_receipt_closes_module_callable_and_executor(self):
        wrapper = batch_wrapper(FakeJitAutoResolver())
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        binding = runtime.jit_executor_binding

        binding.validate(runtime.jit_module_binding, runtime.operator_session.callable_binding)
        self.assertIs(binding.executor, runtime._executor)
        self.assertNotIn("executor", binding.to_dict())
        self.assertEqual(len(binding.fingerprint), 64)

    def test_missing_binder_stops_before_jit_or_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(components["events"], environment)
        artifact_resolver = RecordingJitArtifactResolver(
            components["events"], jit_resolver
        )
        module_resolver = RecordingJitModuleResolver(
            components["events"], jit_resolver
        )
        implementation = AttentionOperatorPackageRuntimeImplementation(
            priority=base.priority,
            package_resolver=base._package_resolver,
            plan_gate=components["gate"],
            authority_resolver=FakeJitAuthorityResolver(
                components["events"], environment
            ),
            logical_factory=components["factory"],
            logical_run_adapter=FakeLogicalRunAdapter(),
            tensor_materializer=components["materializer"],
            jit_plan_resolver=jit_resolver,
            jit_artifact_resolver=artifact_resolver,
            jit_module_resolver=module_resolver,
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT executor binder"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_binding_happens_after_module_load_and_callable_resolution(self):
        components, implementation, _, _, _ = package_jit_implementation()
        resolved = implementation.resolve(framework_plan(), "npu:0")
        events = components["events"]

        self.assertLess(events.index("module_load"), events.index("resolve_callable"))
        self.assertLess(events.index("resolve_callable"), events.index("executor_bind"))
        self.assertIs(resolved.executor, resolved.jit_executor_binding.executor)
        self.assertEqual(len(components["jit_executor_binder"].calls), 1)

    def test_stale_binder_receipt_fails_before_plan_publication(self):
        resolver = FakeJitAutoResolver()
        resolver.executor_binding_mode = "stale_callable"
        wrapper = batch_wrapper(resolver)

        with self.assertRaisesRegex(SchemaError, "executor binding is stale"):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        self.assertFalse(wrapper._operator_runtime.is_planned)
        self.assertEqual(resolver.executors, [])

    def test_public_plan_owns_binding_and_run_does_not_rebind(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        binding = runtime.jit_executor_binding

        self.assertEqual(
            runtime.operator_session.active_plan.jit_executor_binding_fingerprint,
            binding.fingerprint,
        )
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertIs(runtime.jit_executor_binding, binding)
        self.assertEqual(len(resolver.executor_bindings), 1)

    def test_failed_binding_replan_preserves_previous_runtime(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_binding = runtime.jit_executor_binding
        old_executor = resolver.executors[0]

        resolver.executor_binding_mode = "stale_callable"
        with self.assertRaises(SchemaError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_executor_binding, old_binding)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_executor_binding_drift_before_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = resolver.executors[0]

        other = batch_wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_executor_binding = (
            other._operator_runtime.jit_executor_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "executor active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_non_jit_runtime_rejects_executor_binding(self):
        resolver = FakeJitAutoResolver()
        resolved = resolver.resolve(framework_plan(), "npu:0")
        aot_receipt = replace(resolved.receipt, backend=Backend.ASCENDC_AOT)
        aot_selection = replace(
            resolved.selection,
            dispatch_receipt_fingerprint=aot_receipt.fingerprint,
            backend=Backend.ASCENDC_AOT,
        )
        with self.assertRaisesRegex(SchemaError, "non-JIT.*JIT bindings"):
            replace(
                resolved,
                receipt=aot_receipt,
                selection=aot_selection,
                jit_plan_binding=None,
                jit_artifact_binding=None,
                jit_module_binding=None,
            )

    def test_public_surface_has_no_binder_or_executor_handle(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("binder", parameters)
            self.assertNotIn("executor", parameters)
        self.assertFalse(hasattr(flashinfer.jit, "default_executor_binder"))


from tests.framework.test_attention_jit_active_plan import FakeJitAuthorityResolver, FakeJitAutoResolver, RecordingJitArtifactResolver, RecordingJitModuleResolver, RecordingJitPlanResolver, SyntheticJitExecutorBinder, fake_operation, package_jit_implementation


class AttentionJitPlannerBindingTests(unittest.TestCase):
    """The loaded JIT plan symbol remains private and wrapper-owned."""

    def test_exact_planner_receipt_closes_module_and_plan_factory(self):
        wrapper = batch_wrapper(FakeJitAutoResolver())
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        binding = runtime.jit_planner_binding

        binding.validate(runtime.jit_module_binding)
        self.assertEqual(
            binding.operation_id,
            runtime.operator_session.active_plan.prepared_plan.implementation_id,
        )
        self.assertNotIn("factory", binding.to_dict())
        self.assertEqual(len(binding.fingerprint), 64)

    def test_missing_planner_binder_stops_before_jit_or_package_import(self):
        components = build_components()
        environment = runtime_environment()
        base = components["implementation"]
        jit_resolver = RecordingJitPlanResolver(components["events"], environment)
        artifact_resolver = RecordingJitArtifactResolver(
            components["events"], jit_resolver
        )
        module_resolver = RecordingJitModuleResolver(
            components["events"], jit_resolver
        )
        executor_binder = SyntheticJitExecutorBinder(
            "cann", fake_operation().operation_id, components["events"]
        )
        implementation = AttentionOperatorPackageRuntimeImplementation(
            priority=base.priority,
            package_resolver=base._package_resolver,
            plan_gate=components["gate"],
            authority_resolver=FakeJitAuthorityResolver(
                components["events"], environment
            ),
            logical_factory=components["factory"],
            logical_run_adapter=FakeLogicalRunAdapter(),
            tensor_materializer=components["materializer"],
            jit_plan_resolver=jit_resolver,
            jit_artifact_resolver=artifact_resolver,
            jit_module_resolver=module_resolver,
            jit_executor_binder=executor_binder,
        )
        with self.assertRaisesRegex(
            AttentionOperatorIntegrationError, "JIT planner binder"
        ):
            implementation.resolve(framework_plan(), "npu:0")
        self.assertEqual(jit_resolver.calls, 0)
        self.assertNotIn("resolve_callable", components["events"])

    def test_planner_binding_precedes_callable_and_executor_binding(self):
        components, implementation, _, _, _ = package_jit_implementation()
        resolved = implementation.resolve(framework_plan(), "npu:0")
        events = components["events"]

        self.assertLess(events.index("module_load"), events.index("planner_bind"))
        self.assertLess(events.index("planner_bind"), events.index("resolve_callable"))
        self.assertLess(events.index("resolve_callable"), events.index("executor_bind"))
        self.assertIs(resolved.factory, resolved.jit_planner_binding.factory)
        self.assertEqual(len(components["jit_planner_binder"].calls), 1)

    def test_stale_planner_receipt_fails_before_plan_publication(self):
        resolver = FakeJitAutoResolver()
        resolver.planner_binding_mode = "stale_module"
        wrapper = batch_wrapper(resolver)

        with self.assertRaisesRegex(SchemaError, "planner binding is stale"):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        self.assertFalse(wrapper._operator_runtime.is_planned)
        self.assertEqual(resolver.executors, [])

    def test_public_plan_owns_planner_binding_without_rebinding_on_run(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        binding = runtime.jit_planner_binding

        self.assertEqual(
            runtime.operator_session.active_plan.jit_planner_binding_fingerprint,
            binding.fingerprint,
        )
        wrapper.run("q1", ("k", "v"))
        wrapper.run("q2", ("k", "v"))
        self.assertIs(runtime.jit_planner_binding, binding)
        self.assertEqual(len(resolver.planner_bindings), 1)

    def test_failed_planner_replan_preserves_previous_runtime(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        runtime = wrapper._operator_runtime
        old_plan = wrapper.plan_state
        old_binding = runtime.jit_planner_binding
        old_executor = resolver.executors[0]

        resolver.planner_binding_mode = "stale_module"
        with self.assertRaises(SchemaError):
            plan_public_wrapper(wrapper, page_size=16, kv_lengths=(32, 16))
        self.assertIs(wrapper.plan_state, old_plan)
        self.assertIs(runtime.jit_planner_binding, old_binding)
        wrapper.run("old", ("k", "v"))
        self.assertEqual(len(old_executor.calls), 1)

    def test_run_detects_planner_binding_drift_before_executor(self):
        resolver = FakeJitAutoResolver()
        wrapper = batch_wrapper(resolver)
        plan_public_wrapper(wrapper, page_size=16, kv_lengths=(24, 16))
        executor = resolver.executors[0]

        other = batch_wrapper(FakeJitAutoResolver())
        plan_public_wrapper(other, page_size=16, kv_lengths=(32, 16))
        wrapper._operator_runtime._jit_planner_binding = (
            other._operator_runtime.jit_planner_binding
        )
        with self.assertRaisesRegex(AttentionStateError, "planner active-plan"):
            wrapper.run("q", ("k", "v"))
        self.assertEqual(executor.calls, [])

    def test_non_jit_runtime_rejects_planner_binding(self):
        resolver = FakeJitAutoResolver()
        resolved = resolver.resolve(framework_plan(), "npu:0")
        aot_receipt = replace(resolved.receipt, backend=Backend.ASCENDC_AOT)
        aot_selection = replace(
            resolved.selection,
            dispatch_receipt_fingerprint=aot_receipt.fingerprint,
            backend=Backend.ASCENDC_AOT,
        )
        with self.assertRaisesRegex(SchemaError, "non-JIT.*JIT bindings"):
            replace(
                resolved,
                receipt=aot_receipt,
                selection=aot_selection,
                jit_plan_binding=None,
                jit_artifact_binding=None,
                jit_module_binding=None,
                jit_executor_binding=None,
            )

    def test_public_surface_has_no_planner_or_jit_handle(self):
        for callable_value in (BatchAttention, BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("planner", parameters)
            self.assertNotIn("jit", parameters)
        self.assertFalse(hasattr(flashinfer.jit, "default_planner_binder"))
