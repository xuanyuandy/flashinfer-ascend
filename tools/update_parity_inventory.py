"""Inventory declarations without importing the CUDA upstream checkout.

This does not replace the missing 147-row CANN/consumer coverage matrix. Unknown
coverage stays unverified. Running twice is idempotent for a fixed upstream ref.
"""

import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess


REF = "2895577720741ddfedff118202c89584996b8875"
LOCAL_COVERAGE_EVIDENCE = "当前工作区实现与 work_items/reports/execution/steps 中的真实 A5 验证；不代替原始覆盖表。"
DOMAINS = {
    "activation", "attention", "cascade", "decode", "fused_moe", "gemm",
    "mla", "norm", "page", "prefill", "quantization", "rope", "sampling",
    "sparse", "tllm_enums",
}
ATTENTION_CLASSES = (
    "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
    "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
)
# Mappings established by the recorded implementations and A5 numerical tests.
# No CANN support is inferred merely from an upstream function's name.
CANN_OPERATORS = {
    "flashinfer.norm.rmsnorm": ["npu_rms_norm"],
    "flashinfer.norm.fused_add_rmsnorm": ["npu_add_rms_norm"],
    "flashinfer.norm.gemma_rmsnorm": ["npu_gemma_rms_norm"],
    "flashinfer.norm.layernorm": ["aten.layer_norm (PrivateUse1/ACLNN)"],
    "flashinfer.rope.apply_rope_with_cos_sin_cache_inplace": ["npu_rotary_mul"],
    "flashinfer.gemm.bmm_fp8": ["npu_grouped_matmul"],
    "flashinfer.gemm.gemm_fp8_nt_groupwise": ["npu_grouped_matmul"],
    "flashinfer.fused_moe.cutlass_fused_moe": ["npu_grouped_matmul"],
    "flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe": ["npu_grouped_matmul"],
    "flashinfer.fused_moe.trtllm_fp8_block_scale_moe": ["npu_grouped_matmul"],
    "flashinfer.cascade.merge_state": ["npu_attention_update"],
    "flashinfer.cascade.merge_states": ["npu_attention_update"],
    "flashinfer.page.append_paged_kv_cache": ["npu_scatter_pa_kv_cache"],
    "flashinfer.activation.silu_and_mul": ["npu_swiglu"],
    "flashinfer.activation.gelu_and_mul": ["npu_gelu_mul"],
    "flashinfer.activation.gelu_tanh_and_mul": ["npu_gelu_mul"],
}
SAMPLING = (
    "sampling_from_probs sampling_from_logits top_p_sampling_from_probs "
    "top_k_sampling_from_probs top_k_top_p_sampling_from_probs "
    "min_p_sampling_from_probs top_p_renorm_probs top_k_renorm_probs "
    "top_k_mask_logits chain_speculative_sampling"
).split()
for _name in SAMPLING:
    CANN_OPERATORS["flashinfer.sampling." + _name] = (
        [] if _name.endswith("renorm_probs") or _name == "top_k_mask_logits"
        else ["npu_top_k_top_p_sample"]
    )
for _symbol in (
    "flashinfer.prefill.single_prefill_with_kv_cache",
    "flashinfer.decode.single_decode_with_kv_cache",
    "flashinfer.decode.cudnn_batch_decode_with_kv_cache",
    *ATTENTION_CLASSES,
):
    CANN_OPERATORS[_symbol] = ["npu_fused_infer_attention_score"]


class Upstream:
    def __init__(self, root):
        self.root = str(root)
        self.paths = set(self.git("ls-tree", "-r", "--name-only", REF).splitlines())

    def git(self, *args):
        return subprocess.check_output(["git", "-C", self.root, *args], text=True)

    @lru_cache(None)
    def module(self, name):
        stem = name.replace(".", "/")
        path = next((p for p in (stem + ".py", stem + "/__init__.py")
                     if p in self.paths), None)
        if path is None:
            return None
        source = self.git("show", REF + ":" + path)
        return path, source, ast.parse(source)

    def declaration(self, module, attrs, seen=()):
        key = (module, tuple(attrs))
        if key in seen or self.module(module) is None:
            return None
        path, source, tree = self.module(module)
        namespace = tree
        for index, name in enumerate(attrs):
            definitions = [n for n in namespace.body if
                           isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                           and n.name == name]
            if definitions:
                namespace = definitions[-1]  # Implementation after @overload stubs.
                if index == len(attrs) - 1:
                    return path, source, namespace, module + "." + ".".join(attrs)
                if not isinstance(namespace, ast.ClassDef):
                    return None
                continue
            for node in namespace.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets
                ):
                    if isinstance(node.value, ast.Name):
                        return self.declaration(module, attrs[:index] + [node.value.id] + attrs[index+1:], seen + (key,))
                    if index == len(attrs) - 1:
                        return path, source, node, module + "." + ".".join(attrs)
            if index:
                return None
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                package = module.split(".") if path.endswith("/__init__.py") else module.split(".")[:-1]
                base = (package[:len(package) - node.level + 1]
                        if node.level else [])
                target = ".".join(base + ((node.module or "").split(".") if node.module else []))
                for alias in node.names:
                    if (alias.asname or alias.name) == name or alias.name == "*":
                        result = self.declaration(
                            target, [name if alias.name == "*" else alias.name] + attrs[1:], seen + (key,),
                        )
                        if result is not None:
                            return result
            return None

    def resolve(self, symbol):
        parts = symbol.split(".")
        for split in range(len(parts) - 1, 0, -1):
            if self.module(".".join(parts[:split])):
                return self.declaration(".".join(parts[:split]), parts[split:])
        return None

    def exported_symbols(self):
        root = self.module("flashinfer")[2]
        for node in root.body:
            if (isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
                    and node.module.split(".")[0] in DOMAINS):
                for alias in node.names:
                    symbol = "flashinfer." + node.module + "." + alias.name
                    resolved = self.resolve(symbol)
                    if resolved is None:
                        raise ValueError("cannot resolve upstream export: " + symbol)
                    yield symbol
                    if isinstance(resolved[2], ast.ClassDef):
                        for child in resolved[2].body:
                            if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
                                yield symbol + "." + child.name


def annotate(entry, upstream):
    symbol = entry["upstream"]
    resolved = upstream.resolve(symbol)
    if resolved is None:
        if symbol.endswith(".plan_selection"):
            kind = "npu_extension"
            reason = "NPU 扩展：查询已解析的执行选择及登记表版本；固定上游没有同名公开属性。"
        elif symbol in {
            "flashinfer.runtime.schema.ArtifactRef",
            "flashinfer.runtime.schema.KernelDescriptor",
            "flashinfer.runtime.schema.SourceRef",
        }:
            kind = "npu_extension"
            reason = "NPU 扩展：登记 Ascend 执行体、源码及 SOC 编译目标；固定上游没有本仓 runtime 静态契约。"
        elif symbol in {
            "flashinfer.runtime.schema.CapabilitySpec",
            "flashinfer.runtime.schema.ShapeRange",
        }:
            kind = "npu_extension"
            reason = "NPU 扩展：在 dispatch 选型阶段声明并解释 kernel 的输入规模限制；固定上游没有本仓 runtime 能力区间契约。"
        elif symbol == "flashinfer.runtime.hardware.HardwareSignature":
            kind = "npu_extension"
            reason = "NPU 扩展：固定上游没有本仓 Ascend 硬件签名及 UB 探测契约。"
        elif symbol in {
            "flashinfer.runtime.signature.InputSignature",
            "flashinfer.runtime.signature.build_signature",
        }:
            kind = "npu_extension"
            reason = "NPU 扩展：固定上游没有本仓运行时输入签名及 Ascend 物理排布查询契约。"
        elif symbol.endswith(".end_forward") and any(symbol.startswith(c + ".") for c in ATTENTION_CLASSES):
            kind = "compatibility_extension"
            reason = "保留旧版 end_forward 空操作供历史调用兼容；指定上游提交已无该方法，不计入其覆盖分母。"
        else:
            raise ValueError("unresolved ledger symbol: " + symbol)
        entry.update(api_kind=kind, semantic_status="intentionally_different", reason=reason)
        entry.pop("upstream_source", None)
    else:
        path, source, node, declaration = resolved
        entry.update(api_kind="upstream", upstream_source=dict(
            commit=REF, path=path, line=node.lineno,
            sha256=hashlib.sha256(source.encode()).hexdigest(), declaration=declaration,
        ))
    entry["local_status"] = entry["implementation_status"]
    entry.setdefault("consumers", {"vllm": "unverified", "sglang": "unverified"})
    owner = symbol if symbol in CANN_OPERATORS else next((
        s for s in ATTENTION_CLASSES
        if symbol in {s + ".run", s + ".forward"}
        and entry["implementation_status"] in {"functional", "optimized"}
    ), None)
    if owner is not None:
        if entry.get("cann_coverage", "unverified") in {"unverified", "verified", "verified_composite"} or entry.get("coverage_evidence") == LOCAL_COVERAGE_EVIDENCE:
            operators = CANN_OPERATORS[owner]
            entry["cann_coverage"] = "covered" if operators else "unverified"
            entry["cann_operators"] = operators
            entry["coverage_evidence"] = LOCAL_COVERAGE_EVIDENCE
    else:
        if entry.get("coverage_evidence") == LOCAL_COVERAGE_EVIDENCE:
            entry.update(cann_coverage="unverified", cann_operators=[])
            entry.pop("coverage_evidence", None)
        entry.setdefault("cann_coverage", "unverified")
        entry.setdefault("cann_operators", [])
    if entry["semantic_status"] == "intentionally_different" and entry["implementation_status"] == "missing":
        entry["cann_coverage"] = "not_applicable"
    if symbol.endswith(".is_cuda_graph_enabled") or symbol.endswith(".use_tensor_cores"):
        entry.update(local=symbol, semantic_status="compatible", reason="",
                     notes="保留上游同名属性作为兼容入口；NPU 图捕获冻结语义仍由 T9 实现。")
    if symbol.endswith(".workspace_size") and any(symbol.startswith(c + ".") for c in ATTENTION_CLASSES):
        entry.update(implementation_status="framework", local_status="framework",
                     notes="Host reference 和外部 provider 的只读 workspace 查询可执行；原生 CANN 最大 workspace 尚待 T10，当前显式报错。")
    if symbol == "flashinfer.fused_moe.RoutingMethodType":
        entry.update(upstream_alias_of="flashinfer.tllm_enums.RoutingMethodType",
                     notes="兼容导入路径；唯一枚举定义为 flashinfer.tllm_enums.RoutingMethodType，11 个成员的值与固定上游一致。")
    if entry["api_kind"] != "upstream":
        entry.update(cann_coverage="not_applicable", cann_operators=[])
    return entry


def refresh(manifest, upstream, all_symbols=()):
    entries = {e["upstream"]: e for e in manifest["entries"]}
    for symbol in all_symbols:
        entries.setdefault(symbol, dict(
            upstream=symbol, local=symbol, upstream_kind="stable",
            semantic_status="unassessed", implementation_status="missing", priority="P2",
            notes="固定上游公开声明的库存条目；未确认 CANN 覆盖和下游使用，不代表 147 条原始覆盖表。",
        ))
    manifest["entries"] = [annotate(entry, upstream) for _, entry in sorted(entries.items())]
    manifest["upstream"]["ref"] = REF
    manifest["generated_at"] = "2026-10-21"
    manifest["inventory_status"] = "bootstrap"
    manifest["coverage_matrix"] = dict(
        status="missing_source", expected_rows=147,
        notes="147 条原始覆盖表、vllm-ascend 检出及下游使用证据尚未提供；当前由指定提交的公开声明补充库存，未核实字段保留 unverified。",
    )
    for surface in manifest.get("attention_surfaces", ()):
        if surface["attention_mode"] != "batch_mixed_paged" and surface["npu_execution"] == "integration_required":
            surface["npu_execution"] = "functional"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/home/dy/projects/flashinfer-ori"))
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    upstream = Upstream(args.upstream)
    manifests = {}
    for name in ("api_parity.json", "attention_api_parity.json"):
        path = Path("flashinfer/data") / name
        raw = (subprocess.check_output(["git", "show", "HEAD:" + str(path)], text=True)
               if args.stage_only else path.read_text())
        manifests[path] = json.loads(raw)
    attention = manifests[Path("flashinfer/data/attention_api_parity.json")]
    general = manifests[Path("flashinfer/data/api_parity.json")]
    # The scoped ledger holds the reviewed Attention Host/provider/native levels.
    general_entries = {e["upstream"]: e for e in general["entries"]}
    for entry in attention["entries"]:
        general_entries.setdefault(entry["upstream"], {}).update(entry)
    general["entries"] = list(general_entries.values())
    refresh(general, upstream, set(upstream.exported_symbols()))
    refresh(attention, upstream)
    counts = {}
    for path, manifest in manifests.items():
        data = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if args.stage_only:
            blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=data, text=True).strip()
            subprocess.run(["git", "update-index", "--cacheinfo", "100644," + blob + "," + str(path)], check=True)
        else:
            path.write_text(data)
        counts[str(path)] = len(manifest["entries"])
    print(json.dumps(dict(commit=REF, entries=counts, coverage_matrix="missing_source"), indent=2))


if __name__ == "__main__":
    main()
