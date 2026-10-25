import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from flashinfer.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    ArtifactVerificationError,
    Backend,
    CompileMode,
    DeviceCapability,
    DispatchError,
    HardwareSignature,
    InputSignature,
    KernelArgumentABI,
    KernelArgumentDirection,
    KernelArgumentPassing,
    KernelBinaryABI,
    KernelDescriptor,
    KernelErrorABI,
    KernelErrorCodeABI,
    KernelLaunchABI,
    KernelRegistry,
    SchemaError,
    SourceRef,
    WorkloadSpec,
    clear_dispatch_cache,
    dispatch,
    get_global_registry,
    load_kernel_manifest,
)
from flashinfer.runtime.manifest import descriptor_from_dict


def artifact(payload=b"synthetic-ascend-object", locator="kernels/test.o"):
    return ArtifactRef(
        kind=ArtifactKind.FILE,
        format=ArtifactFormat.ASCENDC_OBJECT,
        locator=locator,
        digest=hashlib.sha256(payload).hexdigest(),
        target_soc="Ascend910B",
        build_id="test-build-id",
        size_bytes=len(payload),
    )


def launch_abi():
    return KernelLaunchABI(
        abi_name="flashinfer.test.v1",
        entry_point="test_entry",
        argument_names=("x", "out", "workspace", "stream"),
        mutable_arguments=("out", "workspace"),
        stream_argument="stream",
    )


def binary_abi():
    return KernelBinaryABI(
        abi_name="flashinfer.test.binary.v1",
        arguments=(
            KernelArgumentABI(
                "x",
                KernelArgumentPassing.POINTER,
                KernelArgumentDirection.INPUT,
            ),
            KernelArgumentABI(
                "out",
                KernelArgumentPassing.POINTER,
                KernelArgumentDirection.INOUT,
            ),
            KernelArgumentABI(
                "workspace",
                KernelArgumentPassing.POINTER,
                KernelArgumentDirection.INOUT,
                nullable=True,
            ),
            KernelArgumentABI(
                "stream",
                KernelArgumentPassing.OPAQUE_HANDLE,
                KernelArgumentDirection.INPUT,
            ),
        ),
        error_abi=KernelErrorABI(
            "flashinfer.test_error.v1",
            (KernelErrorCodeABI("success", 0),),
        ),
    )


def source():
    return SourceRef(
        language="ascendc",
        locator="csrc/test.cpp",
        digest=hashlib.sha256(b"synthetic source").hexdigest(),
    )


class SourceRefTests(unittest.TestCase):
    def test_construction_and_json_round_trip(self):
        original = source()
        self.assertEqual(original.language, "ascendc")
        self.assertEqual(original.locator, "csrc/test.cpp")
        self.assertEqual(original.schema_version, 1)
        restored = SourceRef.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)

    def test_source_identity_includes_language_path_and_contents(self):
        original = source()
        for changes in (
            dict(language="c++"),
            dict(locator="csrc/other.cpp"),
            dict(digest=hashlib.sha256(b"changed source").hexdigest()),
        ):
            with self.subTest(changes=changes):
                self.assertNotEqual(replace(original, **changes).fingerprint, original.fingerprint)

    def test_invalid_language_path_digest_and_version_are_rejected(self):
        invalid = (
            (dict(language=""), "language"),
            (dict(language="ascend c"), "language"),
            (dict(language=42), "language"),
            (dict(locator=""), "relative POSIX"),
            (dict(locator="."), "relative POSIX"),
            (dict(locator="/tmp/test.cpp"), "relative POSIX"),
            (dict(locator="../test.cpp"), "relative POSIX"),
            (dict(locator="./csrc/test.cpp"), "relative POSIX"),
            (dict(locator="csrc//test.cpp"), "relative POSIX"),
            (dict(locator="csrc\\test.cpp"), "relative POSIX"),
            (dict(locator="csrc/\0.cpp"), "relative POSIX"),
            (dict(locator=42), "relative POSIX"),
            (dict(digest="A" * 64), "SHA-256"),
            (dict(digest="f" * 63), "SHA-256"),
            (dict(digest=None), "SHA-256"),
            (dict(schema_version=2), "schema_version"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    replace(source(), **changes)

    def test_schema_rejects_unknown_and_missing_source_fields(self):
        extra = dict(source().to_dict(), unknown=True)
        missing = source().to_dict()
        missing.pop("digest")
        for value in (extra, missing):
            with self.subTest(value=value):
                with self.assertRaisesRegex(SchemaError, "fields do not match schema"):
                    SourceRef.from_dict(value)


class ArtifactRefTests(unittest.TestCase):
    def test_entry_point_defaults_preserve_legacy_payload_and_fingerprint(self):
        original = artifact()
        self.assertIsNone(original.entry_point)
        legacy_payload = original.to_dict()
        self.assertNotIn("entry_point", legacy_payload)
        self.assertEqual(ArtifactRef.from_dict(legacy_payload), original)
        legacy_fingerprint = hashlib.sha256(json.dumps(
            legacy_payload, sort_keys=True, separators=(",", ":"),
        ).encode("ascii")).hexdigest()
        self.assertEqual(original.fingerprint, legacy_fingerprint)

    def test_object_and_shared_library_entry_points_round_trip(self):
        for fmt in (ArtifactFormat.ASCENDC_OBJECT, ArtifactFormat.SHARED_LIBRARY):
            with self.subTest(format=fmt):
                original = replace(artifact(), format=fmt, entry_point="test_entry")
                restored = ArtifactRef.from_dict(json.loads(json.dumps(original.to_dict())))
                self.assertEqual(restored, original)
                self.assertEqual(restored.to_dict()["entry_point"], "test_entry")
                self.assertNotEqual(original.fingerprint, replace(original, entry_point="other_entry").fingerprint)

    def test_invalid_entry_points_are_rejected(self):
        for entry_point in ("", "test entry", " entry", "entry\0", 42, b"entry"):
            with self.subTest(entry_point=entry_point):
                with self.assertRaisesRegex(SchemaError, "entry_point"):
                    replace(artifact(), entry_point=entry_point)

    def test_round_trip_fingerprint_and_payload_verification(self):
        payload = b"synthetic-ascend-object"
        original = artifact(payload)
        restored = ArtifactRef.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)
        restored.verify_bytes(payload)
        with self.assertRaisesRegex(ArtifactVerificationError, "size mismatch"):
            restored.verify_bytes(payload + b"x")
        with self.assertRaisesRegex(ArtifactVerificationError, "digest mismatch"):
            artifact(bytes([payload[0] ^ 1]) + payload[1:]).verify_bytes(payload)

    def test_file_verification_is_package_root_bounded(self):
        payload = b"verified-object"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "kernels" / "verified.o"
            target.parent.mkdir()
            target.write_bytes(payload)
            reference = artifact(payload, "kernels/verified.o")
            self.assertEqual(reference.verify_file(root), target.resolve())

            outside = root.parent / (root.name + "-outside.o")
            outside.write_bytes(payload)
            link = root / "kernels" / "escape.o"
            try:
                link.symlink_to(outside)
                escaping = artifact(payload, "kernels/escape.o")
                with self.assertRaisesRegex(
                    ArtifactVerificationError, "escapes package root"
                ):
                    escaping.verify_file(root)
            finally:
                outside.unlink()

    def test_locator_kind_format_and_size_are_strict(self):
        base = artifact()
        invalid = (
            (dict(locator="../escape.o"), "normalized relative"),
            (dict(locator="./kernels/test.o"), "normalized relative"),
            (dict(locator="kernels\\test.o"), "normalized relative"),
            (dict(kind=ArtifactKind.BUILTIN), "kind does not match"),
            (dict(size_bytes=None), "requires non-negative size"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    replace(base, **changes)

    def test_builtin_has_provider_digest_but_no_file_bytes(self):
        locator = "builtin:aclnn:attention"
        builtin = ArtifactRef(
            kind=ArtifactKind.BUILTIN,
            format=ArtifactFormat.ACLNN_BUILTIN,
            locator=locator,
            digest=hashlib.sha256(locator.encode("utf-8")).hexdigest(),
            target_soc="Ascend910B",
            build_id="provider-contract-v1",
        )
        with self.assertRaisesRegex(ArtifactVerificationError, "builtin provider"):
            builtin.verify_bytes(b"")


class KernelLaunchABITests(unittest.TestCase):
    def test_round_trip_and_fingerprint(self):
        original = launch_abi()
        restored = KernelLaunchABI.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.fingerprint, original.fingerprint)

    def test_argument_mutation_stream_and_pointer_contracts(self):
        base = launch_abi()
        invalid = (
            (dict(argument_names=("x", "x", "stream")), "unique"),
            (dict(mutable_arguments=("missing",)), "argument subset"),
            (dict(stream_argument="missing"), "must occur"),
            (dict(pointer_width_bits=16), "32 or 64"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    replace(base, **changes)


class ArtifactManifestTests(unittest.TestCase):
    def test_non_reference_descriptor_requires_provenance_and_abi(self):
        with self.assertRaisesRegex(SchemaError, "ArtifactRef"):
            KernelDescriptor(
                "missing-provenance", "test", Backend.ASCENDC_AOT,
                artifact="kernels/test.o",
            )
        with self.assertRaisesRegex(SchemaError, "KernelLaunchABI"):
            KernelDescriptor(
                "missing-abi", "test", Backend.ASCENDC_AOT,
                artifact=artifact(),
            )
        with self.assertRaisesRegex(SchemaError, "KernelBinaryABI"):
            KernelDescriptor(
                "missing-binary-abi", "test", Backend.ASCENDC_AOT,
                artifact=artifact(), launch_abi=launch_abi(),
            )
        with self.assertRaisesRegex(SchemaError, "format does not match"):
            KernelDescriptor(
                "wrong-format", "test", Backend.ACLNN,
                artifact=artifact(), launch_abi=launch_abi(),
                binary_abi=binary_abi(),
            )

    def test_manifest_round_trip_and_bare_string_rejection(self):
        descriptor = KernelDescriptor(
            "structured", "test", Backend.ASCENDC_AOT,
            artifact=artifact(), launch_abi=launch_abi(),
            binary_abi=binary_abi(),
        )
        payload = {
            "schema_version": 3,
            "generated_at": "2026-08-19",
            "kernels": [descriptor.to_dict()],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_kernel_manifest(path), (descriptor,))
            payload["kernels"][0]["artifact"] = "kernels/test.o"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "artifact must be an object"):
                load_kernel_manifest(path)
            payload["schema_version"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "schema_version"):
                load_kernel_manifest(path)


class KernelExecutionBodyTests(unittest.TestCase):
    def tearDown(self):
        clear_dispatch_cache()

    def test_defaults_and_existing_builtin_records_remain_valid(self):
        descriptor = KernelDescriptor("legacy", "test")
        self.assertIsNone(descriptor.artifact)
        self.assertIsNone(descriptor.source)
        self.assertIsNone(descriptor.target_soc)
        self.assertEqual(descriptor.compile_mode, CompileMode.NONE)
        self.assertEqual(descriptor_from_dict(descriptor.to_dict()), descriptor)
        from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID
        from flashinfer.runtime.registry import _builtin_descriptors

        registry = get_global_registry()
        legacy = _builtin_descriptors()
        self.assertEqual(len(legacy), 79)
        self.assertEqual(len(registry.descriptors()), 81)
        self.assertEqual(
            {d.kernel_id for d in registry.descriptors()} - {d.kernel_id for d in legacy},
            {MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID},
        )
        for declaration in legacy:
            existing = registry.get(declaration.kernel_id)
            with self.subTest(kernel_id=existing.kernel_id):
                self.assertEqual(replace(existing), existing)
                self.assertIsNone(existing.source)
                self.assertIsNone(existing.target_soc)
        production = registry.get(MATMUL_BIAS_SILU_KERNEL_ID)
        self.assertEqual(replace(production), production)
        self.assertIsNotNone(production.source)
        self.assertEqual(production.source.language, "ascendc")
        self.assertEqual(production.target_soc, "Ascend950")
        self.assertEqual(production.compile_mode, CompileMode.AOT)
        self.assertEqual(descriptor_from_dict(production.to_dict()), replace(production, runner=None))

    def test_aot_artifact_source_and_target_round_trip_through_manifest(self):
        descriptor = KernelDescriptor(
            "ascendc/test/Ascend910B", "test", Backend.ASCENDC,
            compile_mode="aot", artifact=replace(artifact(), entry_point="test_entry"),
            source=source(), target_soc="Ascend910B",
        )
        payload = {
            "schema_version": 3,
            "generated_at": "2026-10-22",
            "kernels": [descriptor.to_dict()],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored, = load_kernel_manifest(path)
        self.assertEqual(restored, descriptor)
        self.assertEqual(restored.fingerprint, descriptor.fingerprint)
        self.assertEqual(restored.compile_mode, CompileMode.AOT)
        self.assertEqual(restored.artifact.entry_point, "test_entry")

    def test_source_only_jit_needs_no_placeholder_artifact_or_runner(self):
        descriptor = KernelDescriptor(
            "ascendc/test/jit", "test", Backend.ASCENDC,
            compile_mode=CompileMode.JIT, source=source(), target_soc="Ascend950",
        )
        self.assertIsNone(descriptor.artifact)
        self.assertIsNone(descriptor.runner)
        self.assertEqual(descriptor_from_dict(json.loads(json.dumps(descriptor.to_dict()))), descriptor)

    def test_invalid_body_types_target_and_compile_mode_are_rejected(self):
        invalid = (
            (dict(source="csrc/test.cpp"), "SourceRef"),
            (dict(source=source().to_dict()), "SourceRef"),
            (dict(artifact="kernels/test.o"), "ArtifactRef"),
            (dict(target_soc=""), "target_soc"),
            (dict(target_soc="Ascend 950"), "target_soc"),
            (dict(target_soc=950), "target_soc"),
            (dict(compile_mode="invalid"), "compile_mode"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    KernelDescriptor("invalid", "test", Backend.ASCENDC, **changes)

    def test_artifact_requires_an_entry_and_rejects_conflicting_abi_or_target(self):
        with self.assertRaisesRegex(SchemaError, "requires entry_point"):
            KernelDescriptor("missing-entry", "test", Backend.ASCENDC, artifact=artifact())
        with self.assertRaisesRegex(SchemaError, "entry_point does not match"):
            KernelDescriptor(
                "wrong-entry", "test", Backend.ASCENDC_AOT,
                artifact=replace(artifact(), entry_point="other_entry"),
                launch_abi=launch_abi(), binary_abi=binary_abi(),
            )
        with self.assertRaisesRegex(SchemaError, "target_soc does not match"):
            KernelDescriptor(
                "wrong-target", "test", Backend.ASCENDC,
                artifact=replace(artifact(), entry_point="test_entry"), target_soc="Ascend950",
            )

    def test_manifest_rejects_invalid_optional_execution_fields(self):
        base = KernelDescriptor("test", "test", Backend.ASCENDC).to_dict()
        invalid = (
            (dict(source="csrc/test.cpp"), "source must be an object"),
            (dict(source=dict(source().to_dict(), digest="bad")), "SHA-256"),
            (dict(source=dict(source().to_dict(), extra=True)), "fields do not match"),
            (dict(target_soc=950), "target_soc"),
            (dict(compile_mode="invalid"), "compile_mode"),
        )
        for changes, message in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(SchemaError, message):
                    descriptor_from_dict(dict(base, **changes))
        self.assertEqual(
            descriptor_from_dict(dict(base, source=None, target_soc=None)),
            descriptor_from_dict(base),
        )

    def test_execution_contracts_are_tracked_and_survive_ledger_refresh(self):
        from flashinfer.parity import ParityEntry, load_packaged_manifest
        from tools.update_parity_inventory import annotate

        class UpstreamWithoutRuntimeContracts:
            def resolve(self, symbol):
                return None

        entries = {entry.local: entry for entry in load_packaged_manifest("all").entries}
        for name in ("ArtifactRef", "KernelDescriptor", "SourceRef"):
            with self.subTest(name=name):
                entry = entries["flashinfer.runtime.schema." + name]
                refreshed = ParityEntry.from_dict(annotate(
                    entry.to_dict(), UpstreamWithoutRuntimeContracts(),
                ))
                self.assertEqual(refreshed, entry)
                self.assertEqual(entry.api_kind, "npu_extension")
                self.assertEqual(entry.implementation_status, "framework")

    def test_soc_variants_have_distinct_identities_and_select_matching_artifacts(self):
        socs = ("Ascend910B", "Ascend950", "Ascend310P")
        descriptors = tuple(KernelDescriptor(
            "ascendc/test/" + soc, "test", Backend.ASCENDC,
            compile_mode=CompileMode.AOT, source=source(),
            artifact=replace(artifact(), entry_point="test_entry", target_soc=soc),
        ) for soc in socs)
        self.assertEqual(len({d.artifact.fingerprint for d in descriptors}), 3)
        # 保持 kernel_id 一致也必须因为编译目标不同而改变静态身份。
        self.assertEqual(len({replace(d, kernel_id="same-kernel").fingerprint for d in descriptors}), 3)
        registry = KernelRegistry(descriptors)
        for descriptor, soc in zip(descriptors, socs):
            with self.subTest(soc=soc):
                restored = descriptor_from_dict(json.loads(json.dumps(descriptor.to_dict())))
                self.assertEqual(restored, descriptor)
                decision = dispatch(InputSignature("test"), HardwareSignature(soc), registry=registry)
                self.assertEqual(decision.kernel_id, descriptor.kernel_id)
                self.assertEqual(sum(r.accepted for r in decision.reports), 1)
                self.assertEqual(
                    registry.select(WorkloadSpec("test"), DeviceCapability(soc)).kernel_id,
                    descriptor.kernel_id,
                )

    def test_source_only_soc_variants_select_and_explain_target_mismatches(self):
        base = KernelDescriptor(
            "ascendc/source/910b", "test", Backend.ASCENDC,
            compile_mode=CompileMode.JIT, source=source(), target_soc="Ascend910B",
        )
        other = replace(base, kernel_id="ascendc/source/950", target_soc="Ascend950")
        self.assertNotEqual(base.fingerprint, replace(base, target_soc="Ascend950").fingerprint)
        registry = KernelRegistry((base, other))
        decision = dispatch(InputSignature("test"), HardwareSignature("Ascend950"), registry=registry)
        self.assertEqual(decision.kernel_id, other.kernel_id)
        rejected = next(r for r in decision.reports if r.kernel_id == base.kernel_id)
        self.assertFalse(rejected.accepted)
        self.assertIn("target SoC Ascend910B", rejected.reasons[0])
        self.assertEqual(registry.select(WorkloadSpec("test"), DeviceCapability("Ascend950")), other)
        with self.assertRaisesRegex(DispatchError, "target SoC"):
            dispatch(
                InputSignature("test"), HardwareSignature("Ascend950"),
                kernel_id=base.kernel_id, registry=registry,
            )
        with self.assertRaisesRegex(DispatchError, "target SoC"):
            registry.select(WorkloadSpec("test"), DeviceCapability("Ascend310P"))


if __name__ == "__main__":
    unittest.main()
