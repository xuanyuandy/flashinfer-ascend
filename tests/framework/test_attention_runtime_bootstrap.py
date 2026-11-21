"""attention runtime bootstrap regression coverage."""


import unittest

from flashinfer.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionMaterializedOperatorPlanState,
    AttentionMaterializingPlanFactory,
    AttentionMaterializingRunAdapter,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorTensorPlan,
    AttentionOperatorWrapperSession,
    AttentionPlanSpec,
    AttentionMode,
    CannV2PagedPlanFactory,
    CannV2PagedRunAdapter,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedRunAdapter,
    KVLayout,
    MixedPagedKVMetadata,
    bind_attention_operator_callable,
    load_packaged_attention_operator_catalog,
)
from flashinfer.runtime import Backend, SchemaError
from tests.support import hash_value


def framework_plan(provider):
    layout = KVLayout.HND if provider == "cann" else KVLayout.NHD
    page_size = 128 if provider == "cann" else 16
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=layout,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        o_dtype="bfloat16",
    )
    metadata = MixedPagedKVMetadata(
        qo_indptr=(0, 2, 3),
        kv_indptr=(0, 2, 3),
        kv_indices=(7, 3, 9),
        kv_len_arr=(page_size + page_size // 2, page_size),
        page_size=page_size,
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def authority(plan, provider, operation_id, backend):
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.materialization.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_materialization_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-materialization-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-materialization-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=backend,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )
    operation = load_packaged_attention_operator_catalog().get(operation_id)
    probe = AttentionOperatorProviderProbe(
        provider_id=provider,
        adapter_version="checkpoint-016-test",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    selection = AttentionOperatorProviderSelection(
        provider_id=provider,
        provider_probe_fingerprint=probe.fingerprint,
        provider_record_fingerprint=hash_value("9"),
        dispatch_receipt_fingerprint=receipt.fingerprint,
        profile_id=receipt.profile_id,
        profile_fingerprint=receipt.profile_fingerprint,
        backend=receipt.backend,
    )
    observed = AttentionObservedOperatorCallable(
        provider_id=provider,
        package_name=operation.package_name,
        package_version="test-package",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=AttentionObservedCallableSignature(
            operation.positional_arguments,
            operation.keyword_arguments,
            observation_kind="checkpoint-016-test",
        ),
    )
    binding = bind_attention_operator_callable(probe, operation, observed)
    return receipt, selection, binding


class FakeTensor:
    def __init__(self, role, shape, dtype, device, materialization):
        self.role = role
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.materialization = materialization


class FakeMaterializer:
    def __init__(self, provider_id, fail_role=None, return_recipe=False):
        self.provider_id = provider_id
        self.materializer_id = "fake-%s-tensor-materializer-v1" % provider_id
        self.fail_role = fail_role
        self.return_recipe = return_recipe
        self.calls = []

    def materialize(self, tensor_plan, device):
        self.calls.append((tensor_plan, device))
        if tensor_plan.role == self.fail_role:
            raise RuntimeError("fake tensor materialization failure")
        if self.return_recipe:
            return tensor_plan
        return FakeTensor(
            tensor_plan.role,
            tensor_plan.shape,
            tensor_plan.dtype,
            device,
            tensor_plan.materialization,
        )


def components(provider, materializer):
    plan = framework_plan(provider)
    if provider == "cann":
        logical_factory = CannV2PagedPlanFactory()
        logical_adapter = CannV2PagedRunAdapter()
        operation_id = CANN_V2_OPERATION_ID
        backend = Backend.ACLNN
    else:
        logical_factory = FlashAttentionNpuV3PagedPlanFactory()
        logical_adapter = FlashAttentionNpuV3PagedRunAdapter()
        operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID
        backend = Backend.ASCENDC_AOT
    receipt, selection, binding = authority(
        plan, provider, operation_id, backend
    )
    factory = AttentionMaterializingPlanFactory(
        logical_factory, materializer, "npu:0"
    )
    adapter = AttentionMaterializingRunAdapter(logical_adapter)
    return plan, receipt, selection, binding, factory, adapter


def planned_wrapper(provider, materializer):
    plan, receipt, selection, binding, factory, adapter = components(
        provider, materializer
    )
    wrapper = AttentionOperatorWrapperSession(
        load_packaged_attention_operator_catalog()
    )
    wrapper.plan(factory, adapter, plan, receipt, selection, binding)
    return wrapper


def contains_recipe(value):
    if isinstance(value, AttentionOperatorTensorPlan):
        return True
    if isinstance(value, (tuple, list)):
        return any(contains_recipe(item) for item in value)
    return False


class PlanTensorMaterializationTests(unittest.TestCase):
    """Checkpoint 016: plan recipes become provider tensors once, before run."""

    def test_cann_page_table_and_causal_mask_materialize_once_and_reuse(self):
        materializer = FakeMaterializer("cann")
        wrapper = planned_wrapper("cann", materializer)
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertIsInstance(state, AttentionMaterializedOperatorPlanState)
        self.assertEqual(
            tuple(item.role for item in state.tensors),
            ("atten_mask", "block_table"),
        )
        self.assertEqual(len(materializer.calls), 2)

        first = wrapper.run("query-1", ("key-1", "value-1"))
        second = wrapper.run("query-2", ("key-2", "value-2"))
        self.assertEqual(len(materializer.calls), 2)
        first_keywords = dict(first.keyword_arguments)
        second_keywords = dict(second.keyword_arguments)
        self.assertIsInstance(first_keywords["atten_mask"], FakeTensor)
        self.assertIsInstance(first_keywords["block_table"], FakeTensor)
        self.assertIs(
            first_keywords["block_table"], second_keywords["block_table"]
        )
        self.assertFalse(contains_recipe(first.keyword_arguments))

    def test_flash_query_offsets_cache_lengths_and_page_table_materialize(self):
        materializer = FakeMaterializer("flash_attention_npu")
        wrapper = planned_wrapper("flash_attention_npu", materializer)
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertEqual(
            tuple(item.role for item in state.tensors),
            ("cache_seqlens", "cu_seqlens_q", "page_table"),
        )
        lowered = wrapper.run("query", ("key", "value"))
        keywords = dict(lowered.keyword_arguments)
        for name in ("cache_seqlens", "cu_seqlens_q", "page_table"):
            self.assertIsInstance(keywords[name], FakeTensor)
            self.assertEqual(keywords[name].device, "npu:0")
        self.assertFalse(contains_recipe(lowered.keyword_arguments))

    def test_failed_materializing_replan_preserves_previous_runtime(self):
        good = FakeMaterializer("cann")
        wrapper = planned_wrapper("cann", good)
        active = wrapper.active_plan
        runtime = wrapper.runtime_binding

        failing = FakeMaterializer("cann", fail_role="block_table")
        plan, receipt, selection, binding, factory, adapter = components(
            "cann", failing
        )
        with self.assertRaisesRegex(RuntimeError, "materialization failure"):
            wrapper.plan(factory, adapter, plan, receipt, selection, binding)
        self.assertIs(wrapper.active_plan, active)
        self.assertIs(wrapper.runtime_binding, runtime)
        lowered = wrapper.run("query", ("key", "value"))
        self.assertIsInstance(dict(lowered.keyword_arguments)["block_table"], FakeTensor)

    def test_foreign_or_fake_recipe_materializer_output_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "providers differ"):
            AttentionMaterializingPlanFactory(
                CannV2PagedPlanFactory(),
                FakeMaterializer("flash_attention_npu"),
                "npu:0",
            )

        materializer = FakeMaterializer("cann", return_recipe=True)
        plan, receipt, selection, _, factory, _ = components("cann", materializer)
        with self.assertRaisesRegex(SchemaError, "opaque provider tensor"):
            factory.prepare(plan, receipt, selection)


import sys

from flashinfer.attention import (
    EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS,
    AttentionOperatorPackageRuntimeSpec,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorRuntimeResolutionError,
    AttentionOperatorRuntimeResolverRegistry,
    build_attention_operator_package_runtime,
    build_attention_operator_runtime_resolvers,
    build_default_attention_operator_runtime_resolvers,
)
from flashinfer.runtime import SchemaError
from tests.framework.test_attention_capability import bound_kernel, functional_profile, group_plan
from tests.support.attention_provider import FakeLogicalFactory, FakeLogicalRunAdapter, FakePackageLoader, FakePlanGate, FakeTensorMaterializer, fake_operation
from flashinfer.attention import AttentionOperatorOperationCatalog


def bootstrap_components(*, package_version="1.0.0", gate_reasons=()):
    events = []
    operation = fake_operation()
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-022-catalog", operations=(operation,)
    )
    loader = FakePackageLoader(events, version=package_version)
    gate = FakePlanGate(events, gate_reasons)
    profile = functional_profile()
    materializer = FakeTensorMaterializer(events)
    tensor_metadata_inspector = FakeTensorMetadataInspector()
    quantization_binding = AttentionOperatorQuantizationBinding(
        provider_id="cann",
        operation_id=operation.operation_id,
        quant_spec=profile.rules[0].quant_specs[0],
        argument_bindings=(
            AttentionOperatorQuantArgumentBinding(
                "kv.key.scale", "key_scale"
            ),
            AttentionOperatorQuantArgumentBinding(
                "kv.value.scale", "value_scale"
            ),
        ),
    )
    spec = AttentionOperatorPackageRuntimeSpec(
        operation_id=operation.operation_id,
        priority=100,
        adapter_version="checkpoint-022-adapter-v1",
        supported_package_versions=("1.0.0",),
        profiles=(profile,),
        descriptors=(bound_kernel(profile),),
        observed_environment=profile.environment,
        plan_gate=gate,
        logical_factory=FakeLogicalFactory(events),
        logical_run_adapter=FakeLogicalRunAdapter(),
        tensor_materializer=materializer,
        quantization_bindings=(quantization_binding,),
        tensor_metadata_inspector=tensor_metadata_inspector,
    )
    return {
        "events": events,
        "operation": operation,
        "catalog": catalog,
        "loader": loader,
        "gate": gate,
        "materializer": materializer,
        "tensor_metadata_inspector": tensor_metadata_inspector,
        "spec": spec,
    }


class ForeignPlanGate(FakePlanGate):
    provider_id = "flash_attention_npu"


class FakeTensorMetadataInspector:
    """Strict metadata-only inspector used by synthetic package fixtures."""

    def __init__(self):
        self.calls = []

    def to_view(self, tensor, *, name, writable=False):
        self.calls.append((tensor, name, writable))
        view = getattr(tensor, "tensor_view", None)
        if view is None:
            raise SchemaError("synthetic tensor has no TensorView metadata")
        return view


class OperatorRuntimeBootstrapTests(unittest.TestCase):
    """Checkpoint 022: one declarative root composes the NPU runtime tree."""

    def test_default_bootstrap_is_empty_and_import_side_effect_free(self):
        imported_before = set(sys.modules)

        registry = build_default_attention_operator_runtime_resolvers()

        self.assertIs(registry, EMPTY_ATTENTION_OPERATOR_RUNTIME_RESOLVERS)
        self.assertEqual(registry.resolvers, ())
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_build_one_candidate_does_not_probe_or_import_the_package(self):
        values = bootstrap_components()

        implementation = build_attention_operator_package_runtime(
            values["spec"],
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        self.assertEqual(implementation.provider_id, "cann")
        self.assertEqual(implementation.operation_id, values["operation"].operation_id)
        self.assertEqual(implementation.priority, 100)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(values["events"], [])

    def test_registry_routes_npu_and_explain_is_metadata_only(self):
        values = bootstrap_components()
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        self.assertIsInstance(registry, AttentionOperatorRuntimeResolverRegistry)
        self.assertEqual(tuple(item[0] for item in registry.resolvers), ("npu",))
        report = registry.resolvers[0][1].explain(group_plan(), "npu:0")
        self.assertEqual(report.selected.provider_id, "cann")
        self.assertEqual(values["loader"].version_calls, 1)
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertNotIn("authorize", values["events"])
        self.assertNotIn("materialize", values["events"])

    def test_resolve_closes_evidence_before_callable_without_materializing(self):
        values = bootstrap_components()
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        plan = group_plan()

        resolved = registry.resolve(plan, "npu:2")

        self.assertEqual(resolved.framework_plan_fingerprint, plan.fingerprint)
        self.assertEqual(resolved.selection.provider_id, "cann")
        self.assertEqual(resolved.receipt.profile_id, functional_profile().profile_id)
        self.assertEqual(resolved.factory.operation_id, values["operation"].operation_id)
        self.assertEqual(values["loader"].resolve_calls, 1)
        self.assertNotIn("prepare", values["events"])
        self.assertNotIn("materialize", values["events"])

    def test_unavailable_package_stops_before_authority_and_callable(self):
        values = bootstrap_components(package_version=None)
        registry = build_attention_operator_runtime_resolvers(
            (values["spec"],),
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )

        with self.assertRaisesRegex(
            AttentionOperatorRuntimeResolutionError, "not installed"
        ):
            registry.resolve(group_plan(), "npu:0")
        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertNotIn("authorize", values["events"])
        self.assertNotIn("materialize", values["events"])

    def test_catalog_and_component_provider_mismatch_fails_at_build(self):
        values = bootstrap_components()
        spec = values["spec"]
        foreign_gate = ForeignPlanGate(values["events"])
        mismatched = AttentionOperatorPackageRuntimeSpec(
            operation_id=spec.operation_id,
            priority=spec.priority,
            adapter_version=spec.adapter_version,
            supported_package_versions=spec.supported_package_versions,
            profiles=spec.profiles,
            descriptors=spec.descriptors,
            observed_environment=spec.observed_environment,
            plan_gate=foreign_gate,
            logical_factory=spec.logical_factory,
            logical_run_adapter=spec.logical_run_adapter,
            tensor_materializer=spec.tensor_materializer,
            quantization_bindings=spec.quantization_bindings,
            tensor_metadata_inspector=spec.tensor_metadata_inspector,
        )

        with self.assertRaisesRegex(SchemaError, "providers differ"):
            build_attention_operator_package_runtime(
                mismatched,
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["events"], [])

    def test_duplicate_operation_specs_are_rejected_without_package_probe(self):
        values = bootstrap_components()

        with self.assertRaisesRegex(SchemaError, "duplicate.*identity"):
            build_attention_operator_runtime_resolvers(
                (values["spec"], values["spec"]),
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["events"], [])

    def test_spec_rejects_ambiguous_versions_and_tuning_ids(self):
        values = bootstrap_components()
        spec = values["spec"]
        fields = {
            name: getattr(spec, name)
            for name in spec.__dataclass_fields__
            if name != "schema_version"
        }
        cases = (
            (
                {"supported_package_versions": ("1.0.0", "1.0.0")},
                "package versions must be unique",
            ),
            (
                {"tuned_kernel_ids": ("same", "same")},
                "tuned kernel ids must be unique",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                candidate = dict(fields)
                candidate.update(overrides)
                with self.assertRaisesRegex(SchemaError, message):
                    AttentionOperatorPackageRuntimeSpec(**candidate)
