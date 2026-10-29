"""Join static upstream interfaces, current local registrations and reviewed candidates.

The generated table deliberately does not infer numerical equivalence from names.
Hardware/product support and measured coverage remain separate from an API match.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from tools.audit_operator_surface import PUBLIC_MODULES, SourceTree, write_csv


SPECIAL = {
    "attention.single_prefill": ["flashinfer.single_prefill_with_kv_cache"],
    "attention.single_decode": ["flashinfer.single_decode_with_kv_cache"],
    "attention.batch_prefill_paged": ["flashinfer.BatchPrefillWithPagedKVCacheWrapper"],
    "attention.batch_prefill_ragged": ["flashinfer.BatchPrefillWithRaggedKVCacheWrapper"],
    "attention.batch_decode_paged": ["flashinfer.BatchDecodeWithPagedKVCacheWrapper", "flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper"],
    "decode.cudnn_batch_decode_with_kv_cache": ["flashinfer.cudnn_batch_decode_with_kv_cache"],
    "gemm.segment_gemm": ["flashinfer.SegmentGEMMWrapper"],
    "gemm.matmul_bias_silu": ["flashinfer.matmul_bias_silu"],
    "ascendc.single_prefill_catlass": ["flashinfer.single_prefill_with_kv_cache_catlass"],
    "fused_moe.cutlass_fused_moe": ["flashinfer.fused_moe"],
    "fused_moe.trtllm_fp8_block_scale_moe": ["flashinfer.fp8_block_scale_moe"],
    "fused_moe.trtllm_fp8_per_tensor_scale_moe": ["flashinfer.fp8_per_tensor_scale_moe"],
    "attention.decode": [], "attention.prefill": [],
}
UPSTREAM_SPECIAL = {
    "fused_moe.cutlass_fused_moe": ["flashinfer.fused_moe.cutlass_fused_moe"],
    "fused_moe.trtllm_fp8_block_scale_moe": ["flashinfer.fused_moe.trtllm_fp8_block_scale_moe"],
    "fused_moe.trtllm_fp8_per_tensor_scale_moe": ["flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe"],
    "gemm.matmul_bias_silu": [], "ascendc.single_prefill_catlass": [],
}
TORCH_ENTRIES = {
    "norm.rmsnorm": ["torch.ops.flashinfer.rmsnorm"],
    "norm.fused_add_rmsnorm": ["torch.ops.flashinfer.fused_add_rmsnorm"],
    "gemm.matmul_bias_silu": ["torch.ops.flashinfer.matmul_bias_silu", "torch.ops.flashinfer.matmul_bias_silu_out"],
    "ascendc.single_prefill_catlass": ["torch.ops.flashinfer.single_prefill_with_kv_cache_catlass", "torch.ops.flashinfer.single_prefill_with_kv_cache_catlass_out"],
}


def role(row):
    definition = row["definition"]
    if row["call_style"].startswith("wrapper-"):
        return "operator-wrapper"
    if definition["kind"] == "class":
        return "resource-config-or-class"
    if definition["kind"] == "method":
        return "lifecycle-or-method"
    name = row["symbol"].split(".")[-1]
    if name in {"get_batch_indices_positions", "get_seq_lens"}:
        return "metadata-operation"
    if name.startswith(("get_", "is_", "has_", "supports_", "can_", "check_", "validate_", "create_", "destroy_", "free_", "init", "dispose", "reset_", "clear_", "set_", "load_", "save_", "register_", "gen_", "make_", "resolve_", "probe_", "decide_")):
        return "preparation-or-utility"
    if name.startswith(("prepare_", "plan_", "precompile_", "prewarm_", "allocate_", "build_")) or "workspace" in name or name.endswith("_plan"):
        return "preparation-or-utility"
    if any(part in {"autotuner", "autotune_cache", "api_logging", "utils", "jit", "trace", "fi_trace", "profiler", "green_ctx"} for part in definition["canonical"].split(".")):
        return "framework-utility"
    return "compute-or-composition-function"


def family(row):
    text = row["definition"]["canonical"].lower()
    name = row["symbol"].split(".")[-1].lower()
    if role(row) in {"resource-config-or-class", "framework-utility"}:
        return "infrastructure"
    if "gdn" in text or "gated_delta_rule" in text:
        return "linear.gdn"
    if "gdp" in text or "gated_delta_product" in text:
        return "linear.gdp"
    if "kda" in text:
        return "linear.kda"
    if "mamba" in text or "causal_conv" in text or "ssd_combined" in text:
        return "ssm.mamba"
    if ".moe_ep." in text:
        return "moe.ep"
    if ".comm." in text or "parallel_attention" in text:
        return "communication"
    if ".logits_processor." in text:
        return "logits"
    if any(value in text for value in ("diffusion_ops", "minimax", "kimi_", "qwen_", ".mhc", ".msa_ops", "lora", "vision", "deepseek_v41")):
        return "model.specialized"
    if ".mla." in text or "mqa_logits" in text or "mla_" in name:
        return "attention.mla"
    if ".sparse." in text or "block_sparse" in text or "dsa_" in text:
        return "attention.sparse"
    if name == "batchattention":
        return "attention.mixed"
    if ".cascade." in text:
        return "attention.merge" if name.startswith("merge_") else "attention.specialized"
    if ".attention." in text or ".prefill." in text or ".decode." in text or "fmha" in text or ".pod." in text or "xqa" in text:
        if "single_prefill" in name or "single_decode" in name:
            return "attention.single"
        if name in {"batchprefillwithpagedkvcachewrapper", "batchprefillwithraggedkvcachewrapper", "batchdecodewithpagedkvcachewrapper", "cudagraphbatchdecodewithpagedkvcachewrapper", "cudnn_batch_decode_with_kv_cache"}:
            return "attention.batch"
        return "attention.specialized"
    if ".page." in text:
        return "kv"
    if "packbits" in text:
        return "quantization.bits"
    if ".quantization." in text or "quantization" in text:
        return "quantization"
    if ".sampling." in text or "cake_sampling" in text:
        return "sampling"
    if ".topk" in text:
        return "topk"
    if ".rope." in text:
        return "rope"
    if ".norm." in text or "rmsnorm" in text:
        return "norm" if name in {"rmsnorm", "fused_add_rmsnorm", "gemma_rmsnorm", "gemma_fused_add_rmsnorm", "layernorm"} else "norm.fused"
    if ".activation." in text or "gated_act" in text:
        return "activation"
    if ".fused_moe." in text or ".megamoe." in text:
        return "moe.fused"
    if ".gemm." in text or ".grouped_mm." in text or "gemm" in text:
        if "segmentgemmwrapper" in name or "group" in name or "ragged" in name:
            return "gemm.grouped"
        return "gemm.quantized" if any(n in name for n in ("fp8", "fp4", "quant", "mxfp")) else "gemm.dense"
    if "backward" in name or "grad" in name:
        return "training"
    return "infrastructure" if role(row) == "preparation-or-utility" else "model.specialized"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--local-registry", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path("docs/operator-comparison-rules.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    rules = json.loads(args.rules.read_text())
    registry = json.loads(args.local_registry.read_text())
    local = SourceTree(Path(__file__).resolve().parents[1])
    upstream = SourceTree(args.upstream)
    operators = defaultdict(list)
    for record in registry:
        operators[record["op"]].append(record)
    local_rows, canonical_map, local_canonical_ops = [], defaultdict(list), defaultdict(set)
    for op, implementations in sorted(operators.items()):
        names = SPECIAL.get(op, ["flashinfer." + op.split(".")[-1]])
        declarations = [local.resolve(name) for name in names]
        assert all(declarations), (op, names)
        for declaration in declarations:
            local_canonical_ops[declaration["canonical"]].add(op)
        upstream_names = UPSTREAM_SPECIAL.get(op, names)
        for name in upstream_names:
            declaration = upstream.resolve(name)
            if declaration:
                canonical_map[declaration["canonical"]].append((op, names))
        backends = sorted({r["backend"] for r in implementations})
        local_rows.append({"op": op, "public_entries": ";".join(names),
                           "torch_entries": ";".join(TORCH_ENTRIES.get(op, ())),
                           "public_sources": ";".join(d["path"] + ":" + str(d["line"]) for d in declarations),
                           "backends": ";".join(backends),
                           "status": "npu-runner-bound-limited" if any(b != "reference" for b in backends) else "reference-only",
                           "descriptor_count": len(implementations),
                           "kernel_ids": ";".join(r["kernel_id"] for r in implementations),
                           "graph_capabilities": ";".join(sorted({r["graph_capability"] for r in implementations})),
                           "capability_contracts": json.dumps([r["capability"] for r in implementations if r["backend"] != "reference"], separators=(",", ":")),
                           "runner_sources": ";".join(sorted({r["runner_source"]["path"] + ":" + str(r["runner_source"]["line"]) for r in implementations if r.get("runner_source")}))})
    rows, other_rows, canonical = [], [], defaultdict(list)
    for row in inventory["symbols"]:
        if not row["definition"] or row["definition"]["kind"] not in {"function", "class", "method"}:
            continue
        definition = row["definition"]
        group = family(row)
        assert group in rules["groups"]
        matches = canonical_map.get(definition["canonical"], [])
        status = "local-entry-limited" if matches else "no-mapped-local-entry"
        parity_note = "Exact parameter and numerical equivalence is not inferred from declaration matching."
        alternative_entries = []
        if row["symbol"].endswith(".single_prefill_with_kv_cache_return_lse"):
            matches = [("attention.single_prefill", ["flashinfer.single_prefill_with_kv_cache"]) ]
            alternative_entries = ["flashinfer.single_prefill_with_kv_cache(..., return_lse=True)"]
            status = "different-entry-same-computation-candidate"
            parity_note = "The dedicated upstream spelling is absent locally; use the limited local return_lse flag. This does not add an op."
        if role(row) in {"resource-config-or-class", "framework-utility", "preparation-or-utility", "lifecycle-or-method"}:
            status = "auxiliary-separate-review"
        if row["symbol"].endswith("BatchAttention"):
            status = "host-framework-only"
        if "CUDAGraphBatchDecodeWithPagedKVCacheWrapper" in row["symbol"]:
            status = "graph-contract-not-production-capture"
        fields = {"upstream_api": row["symbol"], "canonical_declaration": definition["canonical"], "exposure": row["exposure"],
                  "source": definition["path"] + ":" + str(definition["line"]),
                  "lifecycle": row["call_style"], "role_hint": role(row),
                  "signature": definition.get("signature", definition.get("constructor_signature", "")),
                  "lifecycle_signatures": json.dumps(definition.get("lifecycle_signatures", {}), separators=(",", ":")),
                  "candidate_group": group, "candidate_match": rules["groups"][group]["match"],
                  "official_source_ids": ";".join(rules["groups"][group]["sources"]),
                  "local_status": status,
                  "local_ops": ";".join(sorted({op for op, _ in matches})),
                  "local_entries": ";".join(sorted({name for _, names in matches for name in names})),
                  "alternative_local_entries": ";".join(alternative_entries), "parity_note": parity_note,
                  "static_test_calls": sum(u["usage"] in {"call", "bound-call"} for u in row["usages"]),
                  "test_files": ";".join(sorted({u["path"] for u in row["usages"]})),
                  "numerical_equivalence": "not-established-by-static-scan"}
        if row["exposure"] in {"top-level-export", "facade-module-export"}:
            rows.append(fields)
            canonical[definition["canonical"]].append(fields)
        else:
            other_rows.append(fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "upstream-api-comparison.csv", rows, list(rows[0]))
    write_csv(args.output_dir / "backend-and-experimental-comparison.csv", other_rows, list(rows[0]))
    unresolved = [{"symbol": row["symbol"], "status": "unresolved-static-reference-not-an-operator-support-claim",
                   "test_files": ";".join(sorted({u["path"] for u in row["usages"]})),
                   "call_sites": ";".join(u["path"] + ":" + str(u["line"]) for u in row["usages"] if u["usage"] in {"call", "bound-call"})}
                  for row in inventory["symbols"] if not row["definition"]]
    write_csv(args.output_dir / "unresolved-references.csv", unresolved, ["symbol", "status", "test_files", "call_sites"])
    write_csv(args.output_dir / "local-operators.csv", local_rows, list(local_rows[0]))
    local_api_rows = []
    for symbol in sorted(local.facade_exports()):
        definition = local.resolve(symbol)
        if not definition or definition["kind"] not in {"function", "class"}:
            continue
        ops = local_canonical_ops.get(definition["canonical"], set())
        status = "npu-entry-limited" if ops else "framework-or-helper-not-an-independent-npu-op"
        if symbol.endswith(".BatchAttention"):
            status = "host-framework-only"
        if "CUDAGraphBatchDecodeWithPagedKVCacheWrapper" in symbol:
            status = "graph-contract-not-production-capture"
        local_api_rows.append({"local_api": symbol, "canonical_declaration": definition["canonical"],
                               "source": definition["path"] + ":" + str(definition["line"]),
                               "kind": definition["kind"], "ops": ";".join(sorted(ops)), "status": status})
    write_csv(args.output_dir / "local-public-interfaces.csv", local_api_rows, list(local_api_rows[0]))
    grouped = [{"canonical_declaration": key, "aliases": ";".join(row["upstream_api"] for row in values),
                "lifecycle": values[0]["lifecycle"], "role_hint": values[0]["role_hint"],
                "candidate_group": values[0]["candidate_group"],
                "test_files": ";".join(sorted({name for row in values for name in row["test_files"].split(";") if name}))}
               for key, values in sorted(canonical.items())]
    write_csv(args.output_dir / "canonical-declarations.csv", grouped, list(grouped[0]))
    wrappers = [row for row in rows + other_rows if row["lifecycle"].startswith("wrapper-")]
    write_csv(args.output_dir / "wrapper-lifecycles.csv", wrappers, list(rows[0]))
    summary = {"upstream_commit": inventory["commit"], "upstream_dirty": inventory["dirty"],
               "test_python_files": inventory["summary"]["python_test_files"],
               "static_test_definitions": inventory["summary"]["test_definitions"],
               "raw_symbols": inventory["summary"]["symbols"],
               "raw_unresolved_symbols": inventory["summary"]["call_styles"]["unresolved"],
               "dynamic_lookups": inventory["summary"]["dynamic_lookups"],
               "facade_callable_spellings": len(rows), "canonical_declarations": len(grouped),
               "other_test_callable_spellings": len(other_rows),
               "lifecycle_counts_with_aliases": dict(Counter(r["lifecycle"] for r in rows)),
               "canonical_lifecycle_counts": dict(Counter(r["lifecycle"] for r in grouped)),
               "role_hints_with_aliases": dict(Counter(r["role_hint"] for r in rows)),
               "local_descriptors": len(registry), "local_registered_ops": len(local_rows),
               "local_npu_runner_ops": sum(r["status"] == "npu-runner-bound-limited" for r in local_rows),
               "local_reference_only_ops": sum(r["status"] == "reference-only" for r in local_rows),
               "warning": "Counts include aliases, helpers and resource/config classes; they are not independent mathematical kernel counts. Candidates are not numerical equivalence or hardware verification."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for filename in ("test-file-coverage.csv", "dynamic-lookups.csv"):
        source = args.inventory.parent / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)
    (args.output_dir / "local-descriptors.json").write_text(json.dumps(registry, indent=2) + "\n")
    manifest = {"upstream_commit": inventory["commit"], "upstream_dirty": inventory["dirty"],
                "facade_modules": ["flashinfer", *("flashinfer." + name for name in PUBLIC_MODULES)],
                "raw_inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                "rules_sha256": hashlib.sha256(args.rules.read_bytes()).hexdigest(),
                "local_registry_sha256": hashlib.sha256(args.local_registry.read_bytes()).hexdigest(),
                "upstream_source_sha256": inventory["source_sha256"]}
    (args.output_dir / "scan-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# 当前本仓逻辑 op 清单", "", "由当前运行时登记表生成，按 op 而非公开别名或 descriptor 计数。",
             "`npu-runner-bound-limited` 表示存在受限 NPU runner，不表示所有上游参数、产品或图模式已验证。",
             "完整 dtype/shape/SoC 能力字段与 runner 来源见同目录 CSV/JSON。", "",
             "| op | 顶层公共入口 | backend | 状态 |", "| --- | --- | --- | --- |"]
    for row in local_rows:
        names = "<br/>".join("`" + name + "`" for name in row["public_entries"].split(";") if name) or "内部 reference op"
        lines.append("| `" + row["op"] + "` | " + names + " | " + row["backends"] + " | " + row["status"] + " |")
    lines.extend(["", "`BatchAttention` 是另一个 Host/框架门面，未计入上述生产 NPU op；其默认 NPU provider 尚未接通。",
                  "同一个 op 的 Python 顶层/子模块、Torch functional/out 或 wrapper 生命周期方法不重复计为新 op。", ""])
    (args.output_dir / "local-operators.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
