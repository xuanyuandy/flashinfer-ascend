"""attention provider observation regression coverage."""


import unittest

from flashinfer.attention import (
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorCallableBindingError,
    AttentionOperatorCallableInspector,
    AttentionOperatorProviderProbe,
    bind_attention_operator_callable,
    explain_attention_operator_callable,
    load_packaged_attention_operator_catalog,
    observe_python_callable_signature,
)
from flashinfer.runtime import SchemaError
from tests.support import hash_value


OPERATION_ID = "flash_attention_npu.flash_attn_with_kvcache@v2"


def fake_flash_attn_with_kvcache_v2(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
):
    raise AssertionError("signature fixture must never execute")


def available_probe(version="test-build-v2"):
    return AttentionOperatorProviderProbe(
        provider_id="flash_attention_npu",
        adapter_version="test-adapter",
        available=True,
        package_versions=(("flash-attention-npu", version),),
    )


def observation(signature=None, **changes):
    values = {
        "provider_id": "flash_attention_npu",
        "package_name": "flash-attention-npu",
        "package_version": "test-build-v2",
        "callable_path": "flash_attn.flash_attn_with_kvcache",
        "api_version": "v2",
        "available": True,
        "signature": (
            observe_python_callable_signature(fake_flash_attn_with_kvcache_v2)
            if signature is None
            else signature
        ),
    }
    values.update(changes)
    return AttentionObservedOperatorCallable(**values)


class FakeInspector:
    provider_id = "flash_attention_npu"

    def inspect(self, operation):
        return observation()


class AttentionCallableObservationTests(unittest.TestCase):
    """Checkpoint 011: fake callable inspection, never real package import/call."""

    def test_exact_fake_signature_binds_package_api_and_catalog_identity(self):
        operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)
        observed = observation()
        report = explain_attention_operator_callable(
            available_probe(), operation, observed
        )

        self.assertTrue(report.accepted)
        self.assertEqual(report.reasons, ())
        self.assertEqual(
            observed.signature,
            AttentionObservedCallableSignature.from_dict(
                observed.signature.to_dict()
            ),
        )
        binding = bind_attention_operator_callable(
            available_probe(), operation, observed
        )
        self.assertEqual(binding.operation_id, OPERATION_ID)
        self.assertEqual(binding.api_version, "v2")
        self.assertEqual(binding.package_version, "test-build-v2")
        self.assertEqual(len(binding.fingerprint), 64)
        self.assertIsInstance(FakeInspector(), AttentionOperatorCallableInspector)

    def test_renamed_extra_and_variadic_parameters_cannot_fake_exact_match(self):
        operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)
        exact = observation().signature

        renamed = AttentionObservedCallableSignature(
            exact.positional_arguments,
            tuple(
                "page_table" if item == "block_table" else item
                for item in exact.keyword_arguments
            ),
            observation_kind="test-drift",
        )
        report = explain_attention_operator_callable(
            available_probe(), operation, observation(renamed)
        )
        self.assertIn("observed keyword signature differs from catalog", report.reasons)

        extra = AttentionObservedCallableSignature(
            exact.positional_arguments,
            exact.keyword_arguments + ("debug_option",),
            observation_kind="test-drift",
        )
        with self.assertRaisesRegex(
            AttentionOperatorCallableBindingError, "keyword signature"
        ):
            bind_attention_operator_callable(
                available_probe(), operation, observation(extra)
            )

        variadic = AttentionObservedCallableSignature(
            exact.positional_arguments,
            exact.keyword_arguments,
            has_var_keyword=True,
            observation_kind="test-drift",
        )
        with self.assertRaisesRegex(
            AttentionOperatorCallableBindingError, "variadic callable"
        ):
            bind_attention_operator_callable(
                available_probe(), operation, observation(variadic)
            )

    def test_unavailable_package_version_path_and_api_variant_are_explainable(self):
        operation = load_packaged_attention_operator_catalog().get(OPERATION_ID)

        unavailable_probe = AttentionOperatorProviderProbe(
            provider_id="flash_attention_npu",
            adapter_version="test-adapter",
            available=False,
            unavailable_reasons=("package metadata is missing",),
        )
        unavailable_observation = AttentionObservedOperatorCallable(
            provider_id="flash_attention_npu",
            package_name="flash-attention-npu",
            package_version="unknown",
            callable_path="flash_attn.flash_attn_with_kvcache",
            api_version="v2",
            available=False,
            unavailable_reasons=("callable was not resolved",),
        )
        report = explain_attention_operator_callable(
            unavailable_probe, operation, unavailable_observation
        )
        self.assertIn(
            "provider unavailable: package metadata is missing", report.reasons
        )
        self.assertIn(
            "callable unavailable: callable was not resolved", report.reasons
        )

        version_report = explain_attention_operator_callable(
            available_probe("different-build"), operation, observation()
        )
        self.assertIn(
            "observed package version differs from provider probe",
            version_report.reasons,
        )
        api_report = explain_attention_operator_callable(
            available_probe(),
            operation,
            observation(
                api_version="v3",
                callable_path="flash_attn.v3.flash_attn_with_kvcache",
            ),
        )
        self.assertIn(
            "observed API version does not match catalog operation",
            api_report.reasons,
        )
        self.assertIn(
            "observed callable path does not match catalog operation",
            api_report.reasons,
        )

        with self.assertRaisesRegex(SchemaError, "requires reasons"):
            AttentionObservedOperatorCallable(
                provider_id="flash_attention_npu",
                package_name="flash-attention-npu",
                package_version="unknown",
                callable_path="flash_attn.flash_attn_with_kvcache",
                api_version="v2",
                available=False,
            )


import sys
from dataclasses import replace

from flashinfer.attention import (
    AttentionDispatchError,
    AttentionEvidenceOperatorRuntimeAuthorityResolver,
    AttentionOperatorOperationSpec,
    AttentionOperatorProviderProbe,
    AttentionOperatorRuntimeAuthorityResolver,
)
from flashinfer.runtime import Backend, SchemaError
from tests.framework.test_attention_capability import attention_artifact, bound_kernel, functional_profile, group_plan


def operation(provider_id="cann"):
    plan = group_plan()
    return AttentionOperatorOperationSpec(
        operation_id="%s.checkpoint_021_attention@v1" % provider_id,
        provider_id=provider_id,
        package_name="checkpoint-021-package",
        callable_path="checkpoint_021_package.attention",
        api_version="v1",
        candidate_modes=(plan.spec.mode,),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=("scale",),
        return_names=("output",),
        source_url="https://example.com/checkpoint-021-attention-v1",
    )


def probe(provider_id="cann", version="1.0.0"):
    return AttentionOperatorProviderProbe(
        provider_id=provider_id,
        adapter_version="checkpoint-021-adapter-v1",
        available=True,
        package_versions=(("checkpoint-021-package", version),),
    )


def resolver(*, environment=None, backend="auto", descriptors=None):
    profile = functional_profile()
    return AttentionEvidenceOperatorRuntimeAuthorityResolver(
        operation(),
        (profile,),
        (bound_kernel(profile),) if descriptors is None else descriptors,
        profile.environment if environment is None else environment,
        backend=backend,
    )


class EvidenceAuthorityResolverTests(unittest.TestCase):
    """Checkpoint 021: package authority uses existing evidence dispatch."""

    def test_resolver_implements_authority_protocol_without_package_import(self):
        imported_before = set(sys.modules)
        authority_resolver = resolver()

        self.assertIsInstance(
            authority_resolver, AttentionOperatorRuntimeAuthorityResolver
        )
        self.assertEqual(authority_resolver.provider_id, "cann")
        self.assertEqual(
            authority_resolver.operation_id, operation().operation_id
        )
        self.assertEqual(len(authority_resolver.profile_ids), 1)
        self.assertEqual(len(authority_resolver.kernel_ids), 1)
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

    def test_authorize_closes_plan_evidence_kernel_provider_and_device(self):
        profile = functional_profile()
        descriptor = bound_kernel(profile)
        authority_resolver = AttentionEvidenceOperatorRuntimeAuthorityResolver(
            operation(), (profile,), (descriptor,), profile.environment
        )
        plan = group_plan()
        provider_probe = probe()

        authority = authority_resolver.authorize(
            plan, "npu:3", operation(), provider_probe
        )

        self.assertEqual(authority.framework_plan_fingerprint, plan.fingerprint)
        self.assertEqual(authority.device, "npu:3")
        self.assertEqual(
            authority.provider_probe_fingerprint, provider_probe.fingerprint
        )
        self.assertEqual(authority.operation_fingerprint, operation().fingerprint)
        self.assertEqual(authority.receipt.profile_id, profile.profile_id)
        self.assertEqual(authority.receipt.kernel_id, descriptor.kernel_id)
        self.assertEqual(
            authority.selection.dispatch_receipt_fingerprint,
            authority.receipt.fingerprint,
        )
        self.assertEqual(authority.selection.requested_provider, "cann")
        self.assertEqual(len(authority.fingerprint), 64)
        authority.receipt.validate(
            plan, profile, descriptor, profile.environment
        )

    def test_environment_drift_is_an_evidence_dispatch_rejection(self):
        profile = functional_profile()
        changed = replace(profile.environment, firmware_version="changed")
        authority_resolver = resolver(environment=changed)

        with self.assertRaisesRegex(
            AttentionDispatchError, "environment fingerprint mismatch"
        ):
            authority_resolver.authorize(
                group_plan(), "npu:0", operation(), probe()
            )

    def test_backend_policy_cannot_relabel_an_incompatible_descriptor(self):
        authority_resolver = resolver(backend=Backend.ACLNN)

        with self.assertRaisesRegex(AttentionDispatchError, "backend policy"):
            authority_resolver.authorize(
                group_plan(), "npu", operation(), probe()
            )

    def test_provider_operation_and_device_identity_mismatches_are_rejected(self):
        authority_resolver = resolver()
        plan = group_plan()
        cases = (
            ("cpu", operation(), probe(), "device npu"),
            ("npu:", operation(), probe(), "device npu"),
            ("npu:abc", operation(), probe(), "device npu"),
            (
                "npu:0",
                operation("flash_attention_npu"),
                probe(),
                "different operation",
            ),
            (
                "npu:0",
                operation(),
                probe("flash_attention_npu"),
                "available provider probe",
            ),
        )
        for device, candidate_operation, candidate_probe, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    authority_resolver.authorize(
                        plan, device, candidate_operation, candidate_probe
                    )

    def test_unknown_or_reference_backend_is_rejected_as_schema_input(self):
        for backend, message in (
            ("unknown", "unknown operator authority backend"),
            (Backend.REFERENCE, "cannot request reference"),
        ):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(SchemaError, message):
                    resolver(backend=backend)

    def test_tuned_kernel_can_only_select_an_evidence_accepted_descriptor(self):
        profile = functional_profile()
        default = bound_kernel(profile, priority=20)
        tuned = bound_kernel(
            profile,
            kernel_id="checkpoint_021_tuned_kernel",
            artifact=attention_artifact(
                "artifacts/ascend910b/checkpoint_021_tuned.o"
            ),
            priority=1,
        )
        authority_resolver = AttentionEvidenceOperatorRuntimeAuthorityResolver(
            operation(),
            (profile,),
            (default, tuned),
            profile.environment,
            tuned_kernel_ids=("unknown-kernel", tuned.kernel_id),
        )

        authority = authority_resolver.authorize(
            group_plan(), "npu:0", operation(), probe()
        )

        self.assertEqual(authority.receipt.kernel_id, tuned.kernel_id)
        self.assertEqual(authority.receipt.selection_source, "tuning")


from flashinfer.attention import (
    CANN_V2_OPERATION_ID,
    FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorTensorPlan,
    AttentionOperatorWrapperSession,
    AttentionPlanSpec,
    CannV2PagedPlanFactory,
    CannV2PagedPlanState,
    CannV2PagedRunAdapter,
    FlashAttentionNpuV3PagedPlanFactory,
    FlashAttentionNpuV3PagedPlanState,
    FlashAttentionNpuV3PagedRunAdapter,
    KVLayout,
    PagedKVMetadata,
    PagedPrefillMetadata,
    AttentionMode,
    bind_attention_operator_callable,
    build_attention_dense_page_table,
    load_packaged_attention_operator_catalog,
)
from flashinfer.runtime import Backend, QuantSpec, SchemaError


def framework_plan(*, provider, page_size=None, layout=None, quant=None):
    if provider == "cann":
        page_size = 128 if page_size is None else page_size
        layout = KVLayout.HND if layout is None else layout
        last_page_len = (min(64, page_size), page_size)
    else:
        page_size = 16 if page_size is None else page_size
        layout = KVLayout.NHD if layout is None else layout
        last_page_len = (8, 16)
    kv_dtype = "bfloat16" if quant is None else quant.storage_dtype
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=layout,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype=kv_dtype,
        o_dtype="bfloat16",
        kv_quant_spec=quant,
    )
    metadata = PagedPrefillMetadata(
        qo_indptr=(0, 2, 3),
        paged_kv=PagedKVMetadata(
            indptr=(0, 2, 3),
            indices=(7, 3, 9),
            last_page_len=last_page_len,
            page_size=page_size,
        ),
    )
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def runtime_authority(plan, provider, operation_id, backend):
    receipt = AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="fake.provider.adapter.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="fake_provider_adapter_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="fake-provider-adapter-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-provider-adapter-kernel",
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
        adapter_version="checkpoint-013-test",
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
            observation_kind="checkpoint-013-test",
        ),
    )
    binding = bind_attention_operator_callable(probe, operation, observed)
    return receipt, selection, binding


def planned_wrapper(provider):
    plan = framework_plan(provider=provider)
    if provider == "cann":
        factory = CannV2PagedPlanFactory()
        adapter = CannV2PagedRunAdapter()
        operation_id = CANN_V2_OPERATION_ID
        backend = Backend.ACLNN
    else:
        factory = FlashAttentionNpuV3PagedPlanFactory()
        adapter = FlashAttentionNpuV3PagedRunAdapter()
        operation_id = FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID
        backend = Backend.ASCENDC_AOT
    receipt, selection, binding = runtime_authority(
        plan, provider, operation_id, backend
    )
    wrapper = AttentionOperatorWrapperSession(
        load_packaged_attention_operator_catalog()
    )
    wrapper.plan(factory, adapter, plan, receipt, selection, binding)
    return wrapper


class AttentionProviderAdaptersTests(unittest.TestCase):
    """Checkpoint 013: exact, non-executing provider-specific lowering."""

    def test_csr_page_table_is_densified_once_with_deterministic_padding(self):
        table = build_attention_dense_page_table(
            (0, 2, 3), (7, 3, 9), role="page_table"
        )

        self.assertEqual(table.tensor.shape, (2, 2))
        self.assertEqual(table.row_page_counts, (2, 1))
        self.assertEqual(table.rows, ((7, 3), (9, 0)))
        self.assertEqual(table.tensor.dtype, "int32")
        self.assertEqual(table.tensor.device_policy, "same_as_query")
        self.assertEqual(len(table.fingerprint), 64)

    def test_cann_v2_plan_owns_tnd_lengths_mask_and_exact_call(self):
        imported_before = set(sys.modules)
        wrapper = planned_wrapper("cann")
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertIsInstance(state, CannV2PagedPlanState)
        self.assertEqual(state.query_cumulative_lengths, (2, 3))
        self.assertEqual(state.kv_sequence_lengths, (192, 128))
        self.assertEqual(state.block_table.rows, ((7, 3), (9, 0)))
        self.assertEqual(state.minimum_kv_block_pool_size, 3)
        self.assertEqual(state.sparse_mode, 3)
        self.assertIsInstance(state.causal_mask, AttentionOperatorTensorPlan)
        self.assertEqual(state.causal_mask.materialization, "right_down_causal_2048")

        query, key, value = object(), object(), object()
        lowered = wrapper.run(query, (key, value))
        self.assertEqual(
            lowered.positional_arguments,
            (("query", query), ("key", key), ("value", value)),
        )
        keywords = dict(lowered.keyword_arguments)
        self.assertEqual(keywords["actual_seq_qlen"], [2, 3])
        self.assertEqual(keywords["actual_seq_kvlen"], [192, 128])
        self.assertIs(keywords["block_table"], state.block_table.tensor)
        self.assertIs(keywords["atten_mask"], state.causal_mask)
        self.assertEqual(keywords["input_layout"], "TND")
        self.assertEqual(keywords["block_size"], 128)
        self.assertTrue(keywords["return_softmax_lse"])
        self.assertEqual(lowered.return_names, ("output", "softmax_lse"))
        self.assertEqual(
            lowered.consumed_request_fields,
            ("query", "kv_cache", "return_lse", "logits_soft_cap"),
        )
        imported_after = set(sys.modules).difference(imported_before)
        self.assertNotIn("torch_npu", imported_after)
        self.assertNotIn("flash_attn", imported_after)

        output_only = wrapper.run(query, (key, value), return_lse=False)
        output_only_keywords = dict(output_only.keyword_arguments)
        self.assertFalse(output_only_keywords["return_softmax_lse"])
        self.assertEqual(output_only.return_names, ("output",))

    def test_flash_attention_npu_v3_plan_owns_ragged_and_page_metadata(self):
        wrapper = planned_wrapper("flash_attention_npu")
        state = wrapper.active_plan.prepared_plan.opaque_state
        self.assertIsInstance(state, FlashAttentionNpuV3PagedPlanState)
        self.assertEqual(state.cu_seqlens_q.values, (0, 2, 3))
        self.assertEqual(state.cache_seqlens.values, (24, 16))
        self.assertEqual(state.page_table.rows, ((7, 3), (9, 0)))
        self.assertEqual(state.max_seqlen_q, 2)
        self.assertEqual(state.window_size, (-1, -1))

        query, key, value = object(), object(), object()
        lowered = wrapper.run(query, (key, value))
        self.assertEqual(
            lowered.positional_arguments,
            (("q", query), ("k_cache", key), ("v_cache", value)),
        )
        keywords = dict(lowered.keyword_arguments)
        self.assertIs(keywords["cu_seqlens_q"], state.cu_seqlens_q)
        self.assertIs(keywords["cache_seqlens"], state.cache_seqlens)
        self.assertIs(keywords["page_table"], state.page_table.tensor)
        self.assertEqual(keywords["max_seqlen_q"], 2)
        self.assertTrue(keywords["causal"])
        self.assertEqual(keywords["window_size"], (-1, -1))
        self.assertTrue(keywords["return_softmax_lse"])
        self.assertEqual(lowered.mutable_argument_names, ("k_cache", "v_cache"))

    def test_unverified_layout_quantization_and_run_options_are_rejected(self):
        wrong_layout_plan = framework_plan(provider="cann", layout=KVLayout.NHD)
        receipt, selection, _ = runtime_authority(
            wrong_layout_plan, "cann", CANN_V2_OPERATION_ID, Backend.ACLNN
        )
        with self.assertRaisesRegex(SchemaError, "requires HND"):
            CannV2PagedPlanFactory().prepare(
                wrong_layout_plan, receipt, selection
            )

        wrong_page_plan = framework_plan(provider="cann", page_size=16)
        receipt, selection, _ = runtime_authority(
            wrong_page_plan, "cann", CANN_V2_OPERATION_ID, Backend.ACLNN
        )
        with self.assertRaisesRegex(SchemaError, "block size"):
            CannV2PagedPlanFactory().prepare(wrong_page_plan, receipt, selection)

        quant = QuantSpec(
            scheme="symmetric",
            storage_dtype="int8",
            compute_dtype="bfloat16",
            accumulator_dtype="float32",
        )
        quant_plan = framework_plan(provider="flash_attention_npu", quant=quant)
        receipt, selection, _ = runtime_authority(
            quant_plan,
            "flash_attention_npu",
            FLASH_ATTENTION_NPU_V3_KVCACHE_OPERATION_ID,
            Backend.ASCENDC_AOT,
        )
        with self.assertRaisesRegex(SchemaError, "no verified paged KV quantization"):
            FlashAttentionNpuV3PagedPlanFactory().prepare(
                quant_plan, receipt, selection
            )

        wrapper = planned_wrapper("flash_attention_npu")
        with self.assertRaisesRegex(SchemaError, "KV scale binding"):
            wrapper.run("q", ("k", "v"), k_scale="unverified-scale")
        with self.assertRaisesRegex(SchemaError, "separate"):
            wrapper.run("q", "packed-kv-cache")
