"""Refresh API compatibility annotations from pinned upstream declarations.

Source analysis is independent of CUDA imports. Inventory, implementation and
execution evidence remain separate; unverified annotations stay explicit."""

import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess


REF = "2895577720741ddfedff118202c89584996b8875"
LOCAL_COVERAGE_EVIDENCE = "当前工作区实现与执行期 A5 真机验证（日志为过程产物，未入仓）；不代替原始覆盖表。"
DOMAINS = {
    "activation", "attention", "cascade", "decode", "fused_moe", "gemm",
    "mla", "norm", "page", "prefill", "quantization", "rope", "sampling",
    "sparse", "tllm_enums",
}
ATTENTION_CLASSES = (
    "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
    "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
    "flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper",
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


def declaration_nodes(body):
    """Include version-guarded declarations without entering function scopes."""
    for node in body:
        if isinstance(node, ast.If):
            yield from declaration_nodes(node.body)
            yield from declaration_nodes(node.orelse)
        else:
            yield node


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
            definitions = [n for n in declaration_nodes(namespace.body) if
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


def annotate_attention_workspace(entry):
    """Refresh only T10 evidence, also usable without rebuilding the inventory."""
    symbol = entry["upstream"]
    if not any(symbol == c or symbol.startswith(c + ".") for c in ATTENTION_CLASSES):
        return entry
    if symbol.endswith(".workspace_size"):
        entry.update(implementation_status="framework", local_status="framework",
                     notes="Host reference 和外部 provider 的只读 workspace 查询保留；原生 CANN 显式查询返回 FIA v1 最大字节数，不发布 plan、不写固定 buffer，非图 plan 不执行最大查询。")
    reason = entry.get("reason", "")
    reason = reason.replace("最大 workspace 与重放期动态长度/tiling 更新待 T10/T11", "最大 workspace 已由 use_max_workspace 显式接入；重放期动态长度/tiling 更新待 T11")
    reason = reason.replace("原生最大 workspace 和重放期动态长度/tiling 更新待 T10/T11", "原生最大 workspace 已由 use_max_workspace 显式接入；重放期动态长度/tiling 更新待 T11")
    if "reason" in entry:
        entry["reason"] = reason
    if symbol in ATTENTION_CLASSES:
        entry["notes"] = (entry.get("notes", "").split(" use_max_workspace", 1)[0]
                          + " use_max_workspace=False 为 NPU 扩展的仅关键字参数；显式开启且 use_cuda_graph=True 时，plan 查询 FIA v1 最大字节容量并校验调用方固定 workspace，run 透传同一地址；默认关闭以保留内存代价，未实现的 T11 tiling 更新不在此保证内。")
        entry.setdefault("parameters", {})["use_max_workspace"] = dict(
            api_kind="npu_extension", semantic_status="intentionally_different", default=False,
            extension_namespace="flashinfer.npu", upstream_counterpart=None, drop_in=False,
            reason="T10 NPU-only 最大 workspace 选项；上游无对应参数。保留默认 False；显式查询入口为 flashinfer.npu.maximum_workspace_size。",
        )
    return entry


def annotate(entry, upstream):
    symbol = entry["upstream"]
    resolved = upstream.resolve(symbol)
    if resolved is None:
        if (entry.get("api_kind") == "npu_extension"
                and entry.get("semantic_status") == "intentionally_different"
                and entry.get("reason")):
            kind, reason = "npu_extension", entry["reason"]
        elif symbol in {
            "flashinfer.npu.GraphCapability", "flashinfer.npu.GraphReplaySpec",
            "flashinfer.npu.graph_requirements", "flashinfer.npu.maximum_workspace_size",
            "flashinfer.npu.ReplayParameterUpdates", "flashinfer.npu.ReplayParameterUpdates.recording",
            "flashinfer.npu.ReplayParameterUpdates.before_replay", "flashinfer.npu.ReplayParameterUpdates.tasks",
        }:
            kind = "npu_extension"
            reason = "T10/T11 NPU-only 扩展：descriptor 声明已验证的入图范围、参数与更新策略；调用方拥有 capture/replay 和固定 buffer。固定上游无对应接口，不属于 upstream drop-in API。"
            entry.update(extension_namespace="flashinfer.npu", upstream_counterpart=None, drop_in=False)
        elif symbol == "flashinfer.utils.enable_torch_library":
            kind = "npu_extension"
            reason = "T8 显式、幂等的进程级启用入口；上游无同名入口。保持默认 eager 开销与 torch-free 导入，同时允许在 tracing/capture 之前完成真实注册。"
        elif symbol in {
            "flashinfer.jit.torch_library.AscendCOp",
            "flashinfer.jit.torch_library.matmul_probe_op",
        }:
            kind = "npu_extension"
            reason = "T14 NPU 注册扩展：显式 enable_torch_library 注册 flashinfer:: schema、PrivateUse1 实现与独立 meta；复用 T13 已装载产物，不改变配方、选型或 aclgraph 证据门禁。固定上游无同名接口。"
            entry.update(extension_namespace="flashinfer.jit.torch_library", upstream_counterpart=None, drop_in=False)
        elif symbol.endswith(".plan_selection"):
            kind = "npu_extension"
            reason = "NPU 扩展：查询已解析的执行选择及登记表版本；固定上游没有同名公开属性。"
        elif symbol in {
            "flashinfer.runtime.registry.ImplementationStack",
            "flashinfer.runtime.registry.KernelRegistry.stacks",
            "flashinfer.runtime.registry.KernelRegistry.stack_id_for",
            "flashinfer.runtime.dispatch.StackReport",
        }:
            kind = "npu_extension"
            reason = "NPU 扩展：T19A 按来源及外部仓命名空间分栈，保留 T6/T7 语义并解释栈间与栈内选择；固定上游没有本仓 runtime 两级决策契约。"
        elif symbol in {
            "flashinfer.jit.backends.JitBackend",
            "flashinfer.jit.backends.TritonBackend",
            "flashinfer.jit.backends.AscendCBackend",
            "flashinfer.jit.backends.declare_kernel",
            "flashinfer.jit.cache.JitBuildStore",
            "flashinfer.jit.status.JitArtifactState",
            "flashinfer.jit.status.JitArtifactStatus",
            "flashinfer.jit.status.describe_artifact",
            "flashinfer.jit.build.build_and_load",
            "flashinfer.jit.cache.DiskJitBuildStore",
            "flashinfer.jit.cache.jit_cache_index",
            "flashinfer.jit.loading.DefaultJitModuleLoader",
            "flashinfer.jit.errors.JitBuildError",
            "flashinfer.jit.errors.JitCompilationError",
            "flashinfer.jit.errors.JitLoadError",
            "flashinfer.jit.errors.JitSymbolError",
        }:
            kind = "npu_extension"
            reason = ("NPU 扩展：T13-I 实现惰性 Ascend C host_shared JIT；CATLASS 为固定 submodule 的按需头文件依赖，13-II 的 AOT/op-package 尚未实现。"
                      if symbol == "flashinfer.jit.backends.AscendCBackend" else
                      "NPU 扩展：T18B 实现统一磁盘缓存及 Triton 编译/装载；T13-I 在同一缓存与 loader 上实现 Ascend C host_shared，AOT/op-package 留给 13-II。")
        elif symbol in {
            "flashinfer.jit.recipe.BuildRecipe", "flashinfer.jit.recipe.BuildDependency",
            "flashinfer.jit.recipe.ResolvedBuildRecipe", "flashinfer.jit.recipe.AscendCBuildConfig",
            "flashinfer.jit.recipe.load_build_recipes", "flashinfer.jit.recipe.resolve_build_recipe",
            "flashinfer.jit.ascendc.AscendCToolchain",
        }:
            kind = "npu_extension"
            reason = "T13-I NPU 构建扩展；没有上游同名兼容承诺。只支持 JIT/host_shared，AOT/op-package 留给 13-II。"
        elif symbol in {
            "flashinfer.runtime.schema.GraphCapability",
            "flashinfer.runtime.registry.KernelRegistry.graph_requirements",
        }:
            kind = "npu_extension"
            reason = "NPU 扩展：声明并查询 kernel 的入图性质与待更新参数；固定上游没有本仓 aclgraph 元数据契约。"
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
        if entry.get("cann_coverage", "unverified") in {"unverified", "verified", "verified_composite"} or entry.get("coverage_evidence") in {
            LOCAL_COVERAGE_EVIDENCE, "当前工作区实现与 work_items/reports/execution/steps 中的真实 A5 验证；不代替原始覆盖表。",
        }:
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
        entry.update(local=symbol, semantic_status="compatible", reason="")
        entry.setdefault("notes", "保留上游同名属性作为兼容入口；NPU 图捕获冻结语义仍由 T9 实现。")
    annotate_attention_workspace(entry)
    if symbol == "flashinfer.fused_moe.RoutingMethodType":
        entry.update(upstream_alias_of="flashinfer.tllm_enums.RoutingMethodType",
                     notes="兼容导入路径；唯一枚举定义为 flashinfer.tllm_enums.RoutingMethodType，11 个成员的值与固定上游一致。")
    if entry["api_kind"] != "upstream":
        entry.update(cann_coverage="not_applicable", cann_operators=[])
    return entry


def refresh(manifest, upstream, all_symbols=()):
    entries = {e["upstream"]: e for e in manifest["entries"]}
    for symbol in sorted(all_symbols):
        entries.setdefault(symbol, dict(
            upstream=symbol, local=symbol, upstream_kind="stable",
            semantic_status="unassessed", implementation_status="missing", priority="P2",
            notes="固定上游公开声明的库存条目；未确认 CANN 覆盖和下游使用，不代表 147 条原始覆盖表。",
        ))
    manifest["entries"] = [annotate(entry, upstream) for entry in entries.values()]
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
    parser.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parents[2] / "flashinfer-ori")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    upstream = Upstream(args.upstream)
    manifests = {}
    for name in ("api_parity.json", "attention_api_parity.json"):
        path = Path("tools/metadata") / name
        raw = (subprocess.check_output(["git", "show", "HEAD:" + str(path)], text=True)
               if args.stage_only else path.read_text())
        manifests[path] = json.loads(raw)
    attention = manifests[Path("tools/metadata/attention_api_parity.json")]
    general = manifests[Path("tools/metadata/api_parity.json")]
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
