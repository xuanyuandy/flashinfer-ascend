import unittest

from flashinfer.attention import (
    AttentionOperatorRuntimeImplementationRegistry,
)
from flashinfer.runtime import SchemaError


from tests.support.attention_provider import (
    batch_runtime,
    build_components,
    framework_inputs,
    framework_plan,
    package_attention,
)


class PackageRuntimeIntegrationTests(unittest.TestCase):
    """Checkpoint 019: external package candidates join the atomic runtime."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_missing_package_rejects_without_authority_import_or_materialization(self):
        components = build_components(package_version=None)
        registry = AttentionOperatorRuntimeImplementationRegistry(
            (components["implementation"],)
        )

        report = registry.explain(framework_plan(), "npu:0")

        self.assertFalse(report.candidates[0].accepted)
        self.assertIn("not installed", report.candidates[0].reasons[0])
        self.assertEqual(components["authority"].calls, 0)
        self.assertEqual(components["loader"].resolve_calls, 0)
        self.assertEqual(components["factory"].calls, 0)
        self.assertEqual(components["materializer"].calls, [])

    def test_plan_gate_rejects_without_authority_or_callable_resolution(self):
        components = build_components(gate_reasons=("layout is unsupported",))
        reasons = components["implementation"].rejection_reasons(
            framework_plan(), "npu"
        )

        self.assertEqual(reasons, ("layout is unsupported",))
        self.assertEqual(components["authority"].calls, 0)
        self.assertEqual(components["loader"].resolve_calls, 0)

    def test_authority_is_created_before_callable_import_and_tensor_prepare(self):
        components = build_components()
        resolved = components["implementation"].resolve(
            framework_plan(), "npu:0"
        )

        events = components["events"]
        self.assertLess(events.index("authorize"), events.index("resolve_callable"))
        self.assertNotIn("prepare", events)
        self.assertNotIn("materialize", events)
        self.assertFalse(resolved.executor.is_runtime_bound)

    def test_full_batch_runtime_materializes_once_binds_and_executes_twice(self):
        components = build_components()
        runtime = batch_runtime(components)
        spec, metadata = framework_inputs()

        self.assertIsNone(runtime.plan(spec, metadata))
        first = runtime.run("q1", ("k-cache", "v-cache"))
        second = runtime.run("q2", ("k-cache", "v-cache"))

        self.assertEqual(first, ("package-output:q1", "package-lse:0.25"))
        self.assertEqual(second, ("package-output:q2", "package-lse:0.25"))
        self.assertEqual(len(components["materializer"].calls), 1)
        materialized = components["materializer"].calls[0][2]
        self.assertIs(package_attention.calls[0][3], materialized)
        self.assertIs(package_attention.calls[1][3], materialized)
        self.assertEqual(components["loader"].resolve_calls, 1)
        self.assertEqual(components["authority"].calls, 1)

    def test_failed_replan_authority_preserves_previous_executable_runtime(self):
        components = build_components()
        runtime = batch_runtime(components)
        first_spec, first_metadata = framework_inputs()
        runtime.plan(first_spec, first_metadata)
        first_plan = runtime.plan_state
        first_resolve_count = components["loader"].resolve_calls

        components["authority"].fail = True
        second_spec, second_metadata = framework_inputs(kv_length=64)
        with self.assertRaisesRegex(RuntimeError, "authority failure"):
            runtime.plan(second_spec, second_metadata)

        self.assertIs(runtime.plan_state, first_plan)
        self.assertEqual(components["loader"].resolve_calls, first_resolve_count)
        self.assertEqual(
            runtime.run("old", ("k-cache", "v-cache")),
            ("package-output:old", "package-lse:0.25"),
        )

    def test_stale_authority_is_rejected_before_callable_import(self):
        components = build_components()
        components["authority"].drift = True

        with self.assertRaisesRegex(SchemaError, "pre-import.*stale"):
            components["implementation"].resolve(framework_plan(), "npu:0")
        self.assertEqual(components["loader"].resolve_calls, 0)
        self.assertEqual(components["factory"].calls, 0)


if __name__ == "__main__":
    unittest.main()
