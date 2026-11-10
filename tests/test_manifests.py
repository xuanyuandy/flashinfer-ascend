import json
import tempfile
import unittest
from pathlib import Path

from flashinfer.cli import (
    packaged_attention_capability_manifest_path,
    packaged_kernel_manifest_path,
)
from flashinfer.attention import load_attention_capability_manifest
from flashinfer.parity import (
    ParityManifest,
    load_packaged_manifest,
    packaged_manifest_path,
)
from flashinfer.runtime import SchemaError, load_kernel_manifest


class ManifestTests(unittest.TestCase):
    def test_packaged_kernel_manifest_contains_the_first_production_entry(self):
        from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID
        from flashinfer.runtime import Backend, CompileMode, GraphCapability

        descriptors = load_kernel_manifest(packaged_kernel_manifest_path())
        self.assertEqual(len(descriptors), 3)
        self.assertEqual(
            {item.kernel_id for item in descriptors},
            {MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID,
             "ascendc/gemm/matmul/native/op_package/Ascend950"},
        )
        descriptor = next(item for item in descriptors
                          if item.kernel_id == MATMUL_BIAS_SILU_KERNEL_ID)
        self.assertEqual(descriptor.kernel_id, MATMUL_BIAS_SILU_KERNEL_ID)
        self.assertEqual(descriptor.op, "gemm.matmul_bias_silu")
        self.assertEqual(descriptor.backend, Backend.ASCENDC)
        self.assertEqual(descriptor.compile_mode, CompileMode.AOT)
        self.assertIsNotNone(descriptor.source)
        self.assertEqual(descriptor.target_soc, "Ascend950")
        self.assertEqual(descriptor.launch_abi.entry_point, "aclnnFiMatmulBiasSilu")
        self.assertEqual(descriptor.graph_capability, GraphCapability.NOT_CAPTURABLE)
        self.assertIsNone(descriptor.artifact)
        self.assertIsNone(descriptor.runner)

    def test_packaged_attention_capabilities_are_valid_and_honestly_empty(self):
        self.assertEqual(
            load_attention_capability_manifest(
                packaged_attention_capability_manifest_path()
            ),
            (),
        )

    def test_packaged_parity_manifest_is_valid(self):
        manifest = load_packaged_manifest("all")
        self.assertEqual(manifest.schema_version, 2)
        self.assertEqual(manifest.inventory_status, "bootstrap")
        self.assertGreaterEqual(len(manifest.entries), 147)
        self.assertFalse(manifest.is_complete)
        statuses = {entry.upstream: entry.implementation_status for entry in manifest.entries}
        for symbol in (
            "flashinfer.norm.rmsnorm", "flashinfer.sampling.sampling_from_probs",
            "flashinfer.activation.silu_and_mul", "flashinfer.cascade.merge_states",
            "flashinfer.page.append_paged_kv_cache",
            # T2.3/T2.4 接入后一并纳入已实现集合
            "flashinfer.rope.apply_rope",
            "flashinfer.rope.apply_rope_pos_ids",
            "flashinfer.gemm.mm_bf16",
            "flashinfer.gemm.SegmentGEMMWrapper",
        ):
            self.assertEqual(statuses[symbol], "functional")
        # 仍未接入的标本：llama3.1 的 rope 变体需要额外的频率平滑，不在 T2.3 范围
        self.assertEqual(statuses["flashinfer.rope.apply_llama31_rope"], "missing")
        self.assertEqual(manifest.coverage_matrix["status"], "missing_source")

    def test_attention_parity_inventory_tracks_single_reference_facades(self):
        manifest = load_packaged_manifest("attention")
        self.assertEqual(manifest.schema_version, 2)
        self.assertEqual(manifest.scope, "attention_core")
        self.assertEqual(manifest.inventory_status, "bootstrap")
        self.assertGreaterEqual(len(manifest.entries), 18)
        self.assertEqual(manifest.counts()["missing"], 0)
        statuses = {
            entry.upstream: entry.implementation_status
            for entry in manifest.entries
        }
        for symbol in (
            "flashinfer.attention.BatchAttention",
            "flashinfer.attention.BatchAttention.plan",
            "flashinfer.attention.BatchAttention.run",
        ):
            self.assertEqual(statuses[symbol], "reference")
        self.assertEqual(
            statuses["flashinfer.prefill.single_prefill_with_kv_cache"],
            "functional",
        )
        self.assertEqual(
            statuses["flashinfer.decode.single_decode_with_kv_cache"],
            "functional",
        )
        for symbol in (
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.plan",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.plan",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.forward",
            "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.forward",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.forward",
        ):
            self.assertEqual(statuses[symbol], "functional")
        for symbol in (
            "flashinfer.prefill.single_prefill_with_kv_cache_with_jit_module",
            "flashinfer.decode.single_decode_with_kv_cache_with_jit_module",
            "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.workspace_size",
            "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.workspace_size",
        ):
            self.assertEqual(statuses[symbol], "framework")
        self.assertFalse(manifest.is_complete)

    def test_attention_parity_tracks_the_six_public_provider_routes(self):
        manifest = load_packaged_manifest("attention")
        surfaces = {item.attention_mode: item for item in manifest.attention_surfaces}

        self.assertEqual(
            set(surfaces),
            {
                "single_prefill",
                "single_decode",
                "batch_mixed_paged",
                "batch_prefill_paged",
                "batch_prefill_ragged",
                "batch_decode_paged",
            },
        )
        for mode in ("single_prefill", "single_decode"):
            self.assertEqual(surfaces[mode].public_lifecycle, "one_shot")
            self.assertEqual(surfaces[mode].provider_routing, "ephemeral")
        for mode in (
            "batch_mixed_paged",
            "batch_prefill_paged",
            "batch_prefill_ragged",
            "batch_decode_paged",
        ):
            self.assertEqual(surfaces[mode].public_lifecycle, "plan_run")
            self.assertEqual(surfaces[mode].provider_routing, "mode_bound")
        self.assertTrue(all(item.host_reference for item in surfaces.values()))
        self.assertEqual(surfaces["batch_mixed_paged"].npu_execution, "integration_required")
        self.assertTrue(all(item.npu_execution == "functional" for mode, item in surfaces.items()
                            if mode != "batch_mixed_paged"))
        self.assertEqual(
            manifest.attention_surface_counts(),
            {
                "host_reference": 6,
                "provider_routed": 6,
                "npu_callable": 5,
            },
        )
        report = manifest.report()
        self.assertIn("attention-surfaces: 6", report)
        self.assertIn("provider-routed-surfaces: 6", report)
        self.assertIn("npu-callable-surfaces: 5", report)

    def test_attention_surface_lifecycle_and_inventory_links_are_enforced(self):
        source = json.loads(
            packaged_manifest_path("attention").read_text(encoding="utf-8")
        )
        invalid_values = []

        invalid_lifecycle = json.loads(json.dumps(source))
        invalid_lifecycle["attention_surfaces"][1]["provider_routing"] = (
            "mode_bound"
        )
        invalid_values.append((invalid_lifecycle, "one-shot"))

        unknown_local = json.loads(json.dumps(source))
        unknown_local["attention_surfaces"][0]["local"] = (
            "flashinfer.attention.Unknown"
        )
        invalid_values.append((unknown_local, "not in the inventory"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.json"
            for value, message in invalid_values:
                with self.subTest(message=message):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(SchemaError, message):
                        ParityManifest.load(path)

    def test_duplicate_kernel_ids_are_rejected(self):
        kernel = {
            "kernel_id": "duplicate",
            "op": "rmsnorm",
            "backend": "reference",
        }
        value = {
            "schema_version": 3,
            "generated_at": "2026-08-19",
            "kernels": [kernel, kernel],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "duplicate"):
                load_kernel_manifest(path)


if __name__ == "__main__":
    unittest.main()
