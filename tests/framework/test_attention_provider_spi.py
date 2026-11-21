"""attention provider spi regression coverage."""


import unittest

from flashinfer.attention import (
    ATTENTION_OPERATOR_PROVIDER_VERSION,
    CANN_ATTENTION_PROVIDER_ID,
    FLASH_ATTENTION_NPU_PROVIDER_ID,
    AttentionBackendCapabilityProfile,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionNumericsPolicy,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderRegistry,
    AttentionRuntimeEnvironment,
    AttentionMode,
    KVLayout,
)
from flashinfer.runtime import Backend, SchemaError
from tests.support import hash_value


def draft_profile(profile_id, backend):
    return AttentionBackendCapabilityProfile(
        profile_id=profile_id,
        backend=backend,
        environment=AttentionRuntimeEnvironment(
            soc_version="unknown",
            soc_revision="unknown",
            driver_version="unknown",
            firmware_version="unknown",
            cann_version="unknown",
            torch_version="unknown",
            torch_npu_version="unknown",
            compiler_version="unknown",
            python_abi="unknown",
        ),
        status=AttentionCapabilityStatus.DRAFT,
        numerics_policy=AttentionNumericsPolicy(),
        rules=(
            AttentionCapabilityRule(
                rule_id="paged_prefill_dense_v1",
                modes=(AttentionMode.BATCH_PREFILL_PAGED,),
                kv_layouts=(KVLayout.NHD,),
                dtype_signatures=(("float16", "float16", "float16"),),
            ),
        ),
    )


class FakeProvider:
    def __init__(self, provider_id, probe, profiles):
        self.provider_id = provider_id
        self._probe = probe
        self._profiles = tuple(profiles)
        self.probe_calls = 0

    def probe(self):
        self.probe_calls += 1
        return self._probe

    def capability_profiles(self):
        return self._profiles


class AttentionOperatorProviderSPITests(unittest.TestCase):
    """Checkpoint 004: package discovery and capability ownership only."""

    def test_registration_is_lazy_and_discovery_is_identity_bound(self):
        probe = AttentionOperatorProviderProbe(
            provider_id=CANN_ATTENTION_PROVIDER_ID,
            adapter_version="0.1",
            available=True,
            package_versions=(("torch_npu", "test-only"), ("cann", "test-only")),
        )
        provider = FakeProvider(
            CANN_ATTENTION_PROVIDER_ID,
            probe,
            (draft_profile("cann.test.attention.v1", Backend.ACLNN),),
        )
        registry = AttentionOperatorProviderRegistry((provider,))

        self.assertEqual(provider.probe_calls, 0)
        records = registry.discover()
        self.assertEqual(provider.probe_calls, 1)
        self.assertEqual(records[0].provider_id, CANN_ATTENTION_PROVIDER_ID)
        self.assertEqual(records[0].probe, probe)
        self.assertEqual(len(records[0].fingerprint), 64)
        self.assertEqual(
            AttentionOperatorProviderProbe.from_dict(probe.to_dict()).fingerprint,
            probe.fingerprint,
        )

    def test_unavailable_provider_keeps_explicit_reason_and_declared_profiles(self):
        probe = AttentionOperatorProviderProbe(
            provider_id=FLASH_ATTENTION_NPU_PROVIDER_ID,
            adapter_version="0.1",
            available=False,
            unavailable_reasons=("package is not installed",),
        )
        registry = AttentionOperatorProviderRegistry(
            (
                FakeProvider(
                    FLASH_ATTENTION_NPU_PROVIDER_ID,
                    probe,
                    (
                        draft_profile(
                            "flash_attention_npu.test.attention.v1",
                            Backend.ASCENDC_AOT,
                        ),
                    ),
                ),
            )
        )
        record = registry.discover()[0]

        self.assertFalse(record.probe.available)
        self.assertEqual(
            record.probe.unavailable_reasons, ("package is not installed",)
        )
        self.assertEqual(
            record.profiles[0].profile_id,
            "flash_attention_npu.test.attention.v1",
        )

    def test_duplicate_provider_and_profile_ownership_are_rejected(self):
        self.assertEqual(ATTENTION_OPERATOR_PROVIDER_VERSION, 1)
        cann_probe = AttentionOperatorProviderProbe(
            CANN_ATTENTION_PROVIDER_ID, "0.1", True
        )
        cann = FakeProvider(
            CANN_ATTENTION_PROVIDER_ID,
            cann_probe,
            (draft_profile("shared.attention.v1", Backend.ACLNN),),
        )
        with self.assertRaisesRegex(SchemaError, "duplicate.*provider_id"):
            AttentionOperatorProviderRegistry((cann, cann))

        flash_probe = AttentionOperatorProviderProbe(
            FLASH_ATTENTION_NPU_PROVIDER_ID, "0.1", True
        )
        flash = FakeProvider(
            FLASH_ATTENTION_NPU_PROVIDER_ID,
            flash_probe,
            (draft_profile("shared.attention.v1", Backend.ASCENDC_AOT),),
        )
        with self.assertRaisesRegex(SchemaError, "multiple provider owners"):
            AttentionOperatorProviderRegistry((cann, flash)).discover()


from dataclasses import replace

from flashinfer.attention import (
    CANN_ATTENTION_PROVIDER_ID,
    FLASH_ATTENTION_NPU_PROVIDER_ID,
    AttentionDispatchReceipt,
    AttentionBackendCapabilityProfile,
    AttentionCapabilityRule,
    AttentionCapabilityStatus,
    AttentionNumericsPolicy,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderRecord,
    AttentionOperatorProviderSelectionError,
    AttentionRuntimeEnvironment,
    AttentionMode,
    KVLayout,
    bind_attention_operator_provider,
    explain_attention_operator_provider_binding,
)
from flashinfer.runtime import Backend




def receipt_for(profile):
    return AttentionDispatchReceipt(
        mode=AttentionMode.BATCH_PREFILL_PAGED,
        plan_fingerprint=hash_value("1"),
        admission_fingerprint=hash_value("2"),
        workload_fingerprint=hash_value("3"),
        numerics_policy_fingerprint=profile.numerics_policy.fingerprint,
        profile_id=profile.profile_id,
        profile_fingerprint=profile.fingerprint,
        rule_id=profile.rules[0].rule_id,
        environment_fingerprint=profile.environment.fingerprint,
        evidence_id="fake-provider-binding-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="fake-provider-binding-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=profile.backend,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )


def records():
    cann_profile = draft_profile("cann.test.attention.v1", Backend.ACLNN)
    flash_profile = draft_profile(
        "flash_attention_npu.test.attention.v1", Backend.ASCENDC_AOT
    )
    return (
        AttentionOperatorProviderRecord(
            AttentionOperatorProviderProbe(
                CANN_ATTENTION_PROVIDER_ID,
                "0.1",
                True,
                package_versions=(("torch_npu", "test-only"),),
            ),
            (cann_profile,),
        ),
        AttentionOperatorProviderRecord(
            AttentionOperatorProviderProbe(
                FLASH_ATTENTION_NPU_PROVIDER_ID,
                "0.1",
                False,
                unavailable_reasons=("package is not installed",),
            ),
            (flash_profile,),
        ),
    )


class AttentionProviderBindingTests(unittest.TestCase):
    """Checkpoint 005: receipt-to-provider binding, not operator execution."""

    def test_auto_binds_the_available_owner_and_preserves_full_identity(self):
        provider_records = records()
        receipt = receipt_for(provider_records[0].profiles[0])

        report = explain_attention_operator_provider_binding(
            provider_records, receipt
        )
        self.assertEqual(
            tuple(item.provider_id for item in report.accepted),
            (CANN_ATTENTION_PROVIDER_ID,),
        )
        selection = bind_attention_operator_provider(provider_records, receipt)

        self.assertEqual(selection.provider_id, CANN_ATTENTION_PROVIDER_ID)
        self.assertEqual(selection.dispatch_receipt_fingerprint, receipt.fingerprint)
        self.assertEqual(
            selection.profile_fingerprint, receipt.profile_fingerprint
        )
        self.assertEqual(selection.backend, Backend.ACLNN)
        self.assertEqual(len(selection.fingerprint), 64)

    def test_explicit_provider_policy_is_fail_fast_and_explainable(self):
        provider_records = records()
        receipt = receipt_for(provider_records[0].profiles[0])
        report = explain_attention_operator_provider_binding(
            provider_records,
            receipt,
            provider=FLASH_ATTENTION_NPU_PROVIDER_ID,
        )

        flash = next(
            item
            for item in report.candidates
            if item.provider_id == FLASH_ATTENTION_NPU_PROVIDER_ID
        )
        self.assertIn(
            "provider unavailable: package is not installed", flash.reasons
        )
        self.assertIn(
            "does not own selected profile cann.test.attention.v1", flash.reasons
        )
        with self.assertRaisesRegex(
            AttentionOperatorProviderSelectionError,
            "package is not installed",
        ):
            bind_attention_operator_provider(
                provider_records,
                receipt,
                provider=FLASH_ATTENTION_NPU_PROVIDER_ID,
            )

    def test_stale_profile_and_empty_registry_never_fall_back(self):
        provider_records = records()
        receipt = receipt_for(provider_records[0].profiles[0])
        stale_profile = replace(
            provider_records[0].profiles[0],
            status=provider_records[0].profiles[0].status,
        )
        # Change the profile identity without changing the receipt.
        stale_profile = replace(
            stale_profile,
            rules=(
                replace(stale_profile.rules[0], max_head_dim_qk=128),
            ),
        )
        stale_record = replace(provider_records[0], profiles=(stale_profile,))

        with self.assertRaisesRegex(
            AttentionOperatorProviderSelectionError, "fingerprint is stale"
        ):
            bind_attention_operator_provider(
                (stale_record, provider_records[1]), receipt
            )
        with self.assertRaisesRegex(
            AttentionOperatorProviderSelectionError, "no providers discovered"
        ):
            bind_attention_operator_provider((), receipt)
