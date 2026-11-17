"""attention physical evidence regression coverage."""


import hashlib
import inspect
import unittest
from dataclasses import replace

from flashinfer.attention import (
    AttentionDispatchError,
    AttentionOperatorEvidenceResultArtifact,
    AttentionOperatorPhysicalLayoutEvidence,
    AttentionOperatorPhysicalEvidenceManifest,
    BatchAttention,
    verify_attention_operator_physical_evidence_results,
)
from flashinfer.runtime import SchemaError
from tests.framework.test_package_runtime_integration import package_attention
from tests.framework.test_attention_provider_quantization import active_session
from tests.framework.test_attention_provider_quantization import physical_input, physical_runtime


def digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def physical_evidence(spec, operation, quant_spec, descriptor, **overrides):
    profile = spec.profiles[0]
    kernel = spec.descriptors[0]
    values = {
        "provider_id": operation.provider_id,
        "operation_id": operation.operation_id,
        "evidence_id": "checkpoint-028-physical-functional-v1",
        "runner": "synthetic-physical-layout-harness",
        "suite_id": "checkpoint-028-provider-layout-suite-v1",
        "suite_fingerprint": digest("synthetic-suite-definition"),
        "passed_case_ids": (
            "paged-decode-int8-group-blocked-key",
            "paged-decode-int8-group-blocked-value",
        ),
        "result_digest": digest("synthetic-external-result-record"),
        "quant_spec_fingerprint": quant_spec.fingerprint,
        "descriptor_fingerprint": descriptor.fingerprint,
        "catalog_fingerprint": spec.quant_physical_layout_catalog.fingerprint,
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "rule_id": profile.rules[0].rule_id,
        "environment_fingerprint": profile.environment.fingerprint,
        "kernel_id": kernel.kernel_id,
        "kernel_fingerprint": kernel.fingerprint,
        "artifact_fingerprint": kernel.artifact.fingerprint,
        "launch_abi_fingerprint": kernel.launch_abi.fingerprint,
        "binary_abi_fingerprint": kernel.binary_abi.fingerprint,
    }
    values.update(overrides)
    return AttentionOperatorPhysicalLayoutEvidence(**values)


def evidence_bundle(spec, operation, evidences):
    evidences = tuple(evidences)
    payload = b"synthetic-external-result-record"
    artifacts = tuple(
        AttentionOperatorEvidenceResultArtifact(
            evidence_id=item.evidence_id,
            locator="evidence/%s.result" % item.evidence_id,
            digest=item.result_digest,
            size_bytes=len(payload),
        )
        for item in evidences
    )
    manifest = AttentionOperatorPhysicalEvidenceManifest(
        name="checkpoint-028-synthetic-manifest",
        provider_id=operation.provider_id,
        operation_id=evidences[0].operation_id,
        package_name=operation.package_name,
        adapter_version=spec.adapter_version,
        supported_package_versions=spec.supported_package_versions,
        catalog_fingerprint=spec.quant_physical_layout_catalog.fingerprint,
        evidences=evidences,
        result_artifacts=artifacts,
    )
    return verify_attention_operator_physical_evidence_results(
        manifest,
        {item.locator: payload for item in artifacts},
    )


class PhysicalLayoutEvidenceTests(unittest.TestCase):
    """Checkpoint 028: only exact external physical evidence can authorize."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_evidence_is_canonical_versioned_and_round_trippable(self):
        values, _, spec, quant_spec, descriptor = physical_runtime()
        evidence = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )

        restored = AttentionOperatorPhysicalLayoutEvidence.from_dict(
            evidence.to_dict()
        )

        self.assertEqual(restored, evidence)
        self.assertEqual(restored.fingerprint, evidence.fingerprint)
        self.assertEqual(
            restored.passed_case_ids, tuple(sorted(restored.passed_case_ids))
        )

    def test_schema_requires_real_case_and_digest_identities(self):
        values, _, spec, quant_spec, descriptor = physical_runtime()
        with self.assertRaisesRegex(SchemaError, "passed_case_ids"):
            physical_evidence(
                spec,
                values["operation"],
                quant_spec,
                descriptor,
                passed_case_ids=(),
            )
        with self.assertRaisesRegex(SchemaError, "result_digest"):
            physical_evidence(
                spec,
                values["operation"],
                quant_spec,
                descriptor,
                result_digest="not-a-digest",
            )

    def test_missing_evidence_remains_fail_closed_before_callable_import(self):
        values, plan, spec, _, _ = physical_runtime()

        with self.assertRaisesRegex(
            AttentionDispatchError, "requires exactly one physical-layout evidence"
        ):
            active_session(values, spec=spec, plan=plan)

        self.assertEqual(values["loader"].resolve_calls, 0)
        self.assertEqual(package_attention.calls, [])

    def test_exact_evidence_authorizes_nonlogical_plan_and_metadata_lowering(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        evidence = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        authorized = replace(
            spec,
            physical_layout_evidence_bundle=evidence_bundle(
                spec, values["operation"], (evidence,)
            ),
        )
        active_plan, session = active_session(
            values, spec=authorized, plan=plan
        )
        kv_input = physical_input(quant_spec, descriptor)

        lowered = session.run("query", kv_input)

        receipt = session.active_plan.dispatch_receipt
        self.assertEqual(receipt.evidence_id, evidence.evidence_id)
        self.assertEqual(receipt.evidence_result_digest, evidence.result_digest)
        self.assertEqual(active_plan.fingerprint, plan.fingerprint)
        self.assertIs(dict(lowered.positional_arguments)["key"], kv_input.key_storage)
        self.assertEqual(package_attention.calls, [])

    def test_descriptor_catalog_or_quant_identity_drift_is_rejected(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        base = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        cases = (
            replace(base, descriptor_fingerprint=digest("other-descriptor")),
            replace(base, catalog_fingerprint=digest("other-catalog")),
            replace(base, quant_spec_fingerprint=digest("other-quant")),
        )
        for evidence in cases:
            with self.subTest(field=evidence.fingerprint):
                with self.assertRaises((SchemaError, AttentionDispatchError)):
                    active_session(
                        values,
                        spec=replace(
                            spec,
                            physical_layout_evidence_bundle=evidence_bundle(
                                spec, values["operation"], (evidence,)
                            ),
                        ),
                        plan=plan,
                    )
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_profile_environment_kernel_and_abi_drift_is_rejected(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        base = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        cases = (
            replace(base, profile_fingerprint=digest("other-profile")),
            replace(base, environment_fingerprint=digest("other-environment")),
            replace(base, kernel_fingerprint=digest("other-kernel")),
            replace(base, binary_abi_fingerprint=digest("other-abi")),
        )
        for evidence in cases:
            with self.subTest(field=evidence.fingerprint):
                with self.assertRaises(AttentionDispatchError):
                    active_session(
                        values,
                        spec=replace(
                            spec,
                            physical_layout_evidence_bundle=evidence_bundle(
                                spec, values["operation"], (evidence,)
                            ),
                        ),
                        plan=plan,
                    )
        self.assertEqual(package_attention.calls, [])

    def test_operation_identity_and_duplicate_evidence_are_rejected(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        base = physical_evidence(
            spec, values["operation"], quant_spec, descriptor
        )
        foreign = replace(base, operation_id="foreign.operation")
        with self.assertRaises((SchemaError, AttentionDispatchError)):
            active_session(
                values,
                spec=replace(
                    spec,
                    physical_layout_evidence_bundle=evidence_bundle(
                        spec, values["operation"], (foreign,)
                    ),
                ),
                plan=plan,
            )
        with self.assertRaisesRegex(SchemaError, "evidence ids must be unique"):
            evidence_bundle(spec, values["operation"], (base, base))

    def test_public_flashinfer_plan_run_surface_is_unchanged(self):
        for method in (BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("physical_layout_evidence", parameters)
            self.assertNotIn("provider", parameters)


from flashinfer.attention import (
    AttentionJsonEnvelopeLimits,
    AttentionOperatorEvidenceResultArtifact,
    AttentionOperatorPhysicalEvidenceManifest,
    BatchAttention,
    QuantPhysicalLayoutCatalog,
    load_attention_operator_physical_evidence_manifest,
    verify_attention_operator_physical_evidence_results,
)


RESULT_PAYLOAD = b"synthetic-external-result-record"


def manifest_fixture():
    values, plan, spec, quant_spec, descriptor = physical_runtime()
    evidence = physical_evidence(
        spec, values["operation"], quant_spec, descriptor
    )
    bundle = evidence_bundle(spec, values["operation"], (evidence,))
    return values, plan, spec, quant_spec, descriptor, evidence, bundle


class EvidenceManifestTests(unittest.TestCase):
    """Checkpoint 029: package evidence is bounded JSON plus verified bytes."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_manifest_json_round_trip_is_canonical_and_bounded(self):
        *_, bundle = manifest_fixture()
        encoded = bundle.manifest.to_json()

        restored, usage = load_attention_operator_physical_evidence_manifest(
            encoded
        )

        self.assertEqual(restored, bundle.manifest)
        self.assertEqual(restored.fingerprint, bundle.manifest.fingerprint)
        self.assertEqual(usage.encoded_bytes, len(encoded.encode("utf-8")))
        self.assertGreater(usage.nodes, 1)

    def test_loader_rejects_duplicate_unknown_and_oversized_json(self):
        *_, bundle = manifest_fixture()
        encoded = bundle.manifest.to_json()
        duplicate = encoded[:-1] + ',"name":"duplicate"}'
        with self.assertRaisesRegex(SchemaError, "duplicate JSON object key"):
            load_attention_operator_physical_evidence_manifest(duplicate)

        unknown = encoded[:-1] + ',"unknown":1}'
        with self.assertRaisesRegex(SchemaError, "fields are invalid"):
            load_attention_operator_physical_evidence_manifest(unknown)

        limits = AttentionJsonEnvelopeLimits(max_bytes=len(encoded) - 1)
        with self.assertRaisesRegex(SchemaError, "bytes exceed limit"):
            load_attention_operator_physical_evidence_manifest(
                encoded, limits=limits
            )

    def test_result_locator_is_relative_canonical_and_scoped(self):
        _, _, _, _, _, evidence, _ = manifest_fixture()
        invalid = (
            "/evidence/result.json",
            "../evidence/result.json",
            "evidence/../result.json",
            "evidence//result.json",
            "evidence\\result.json",
            "results/result.json",
        )
        for locator in invalid:
            with self.subTest(locator=locator):
                with self.assertRaisesRegex(SchemaError, "locator"):
                    AttentionOperatorEvidenceResultArtifact(
                        evidence.evidence_id,
                        locator,
                        evidence.result_digest,
                        len(RESULT_PAYLOAD),
                    )

    def test_result_verification_requires_exact_set_size_and_digest(self):
        *_, bundle = manifest_fixture()
        manifest = bundle.manifest
        locator = manifest.result_artifacts[0].locator
        with self.assertRaisesRegex(SchemaError, "payload set"):
            verify_attention_operator_physical_evidence_results(manifest, {})
        with self.assertRaisesRegex(SchemaError, "size mismatch"):
            verify_attention_operator_physical_evidence_results(
                manifest, {locator: RESULT_PAYLOAD[:-1]}
            )
        corrupted = bytearray(RESULT_PAYLOAD)
        corrupted[-1] ^= 1
        with self.assertRaisesRegex(SchemaError, "digest mismatch"):
            verify_attention_operator_physical_evidence_results(
                manifest, {locator: bytes(corrupted)}
            )

    def test_manifest_runtime_identity_closes_package_adapter_and_catalog(self):
        values, _, spec, _, _, _, bundle = manifest_fixture()
        manifest = bundle.manifest
        cases = (
            replace(manifest, adapter_version="other-adapter"),
            replace(manifest, supported_package_versions=("9.9.9",)),
            replace(manifest, package_name="foreign-package"),
        )
        for candidate in cases:
            with self.subTest(candidate=candidate.fingerprint):
                with self.assertRaisesRegex(SchemaError, "runtime identity is stale"):
                    candidate.validate_runtime_spec(
                        values["operation"],
                        spec.adapter_version,
                        spec.supported_package_versions,
                        spec.quant_physical_layout_catalog,
                    )
        with self.assertRaisesRegex(SchemaError, "runtime identity is stale"):
            manifest.validate_runtime_spec(
                values["operation"],
                spec.adapter_version,
                spec.supported_package_versions,
                QuantPhysicalLayoutCatalog(),
            )

    def test_bootstrap_accepts_only_verified_bundle_not_bare_manifest(self):
        _, _, spec, _, _, _, bundle = manifest_fixture()
        with self.assertRaisesRegex(TypeError, "must be a verified bundle"):
            replace(
                spec,
                physical_layout_evidence_bundle=bundle.manifest,
            )

    def test_loaded_and_verified_bundle_authorizes_without_callable_execution(self):
        values, plan, spec, quant_spec, descriptor, _, bundle = manifest_fixture()
        loaded, _ = load_attention_operator_physical_evidence_manifest(
            bundle.manifest.to_json()
        )
        locator = loaded.result_artifacts[0].locator
        verified = verify_attention_operator_physical_evidence_results(
            loaded, {locator: RESULT_PAYLOAD}
        )
        authorized = replace(
            spec, physical_layout_evidence_bundle=verified
        )
        _, session = active_session(values, spec=authorized, plan=plan)

        lowered = session.run("query", physical_input(quant_spec, descriptor))

        self.assertEqual(
            session.active_plan.dispatch_receipt.evidence_id,
            loaded.evidences[0].evidence_id,
        )
        self.assertEqual(lowered.provider_id, "cann")
        self.assertEqual(package_attention.calls, [])

    def test_manifest_remains_absent_from_public_flashinfer_surface(self):
        for method in (BatchAttention.plan, BatchAttention.run):
            parameters = inspect.signature(method).parameters
            self.assertNotIn("evidence_manifest", parameters)
            self.assertNotIn("evidence_bundle", parameters)
