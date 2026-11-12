"""Reproduce the T3 inventory from declarations in the pinned upstream commit."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REF = "2895577720741ddfedff118202c89584996b8875"
SOURCES = {
    "comm.cuda_ipc": (
        "CudaRTLibrary create_shared_buffer free_shared_buffer",
        "依赖 CUDA runtime 的 IPC memory handle；昇腾使用 HCCL/Ascend 内存共享协议，无法复用 CUDA IPC 句柄。",
    ),
    "comm.vllm_ar": (
        "init_custom_ar dispose all_reduce get_graph_buffer_ipc_meta register_buffer register_graph_buffers meta_size",
        "该自定义 all-reduce 绑定 CUDA IPC、GPU 指针及 CUDA graph 注册；HCCL 不兼容其资源和句柄协议。",
    ),
    "comm.trtllm_ar": (
        "AllReduceStrategyType AllReduceStrategyConfig AllReduceFusionOp AllReduceFusionPattern QuantizationSFLayout trtllm_destroy_ipc_workspace_for_all_reduce trtllm_create_ipc_workspace_for_all_reduce_fusion trtllm_destroy_ipc_workspace_for_all_reduce_fusion compute_fp4_swizzled_layout_sf_size trtllm_lamport_initialize trtllm_lamport_initialize_all trtllm_allreduce_fusion trtllm_moe_allreduce_fusion trtllm_moe_finalize_allreduce_fusion",
        "属于 TensorRT-LLM CUDA IPC/Lamport all-reduce 协议或其专属枚举/FP4 scale 排布；不能映射为语义相同的 HCCL 接口。",
    ),
    "comm.trtllm_mnnvl_ar": (
        "MNNVLAllreduceFusionStrategy MNNVLQuantType MNNVLAllReduceFusionWorkspace trtllm_mnnvl_allreduce trtllm_mnnvl_fused_allreduce_add_rmsnorm trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant get_allreduce_mnnvl_workspace trtllm_mnnvl_all_reduce trtllm_mnnvl_fused_allreduce_rmsnorm",
        "依赖 NVIDIA MNNVL/NVLink fabric、multicast 或对称 GPU workspace；昇腾互联与 HCCL 的资源模型不同。",
    ),
    "comm.trtllm_moe_alltoall": (
        "moe_a2a_active_rank_mask moe_a2a_initialize moe_a2a_wrap_payload_tensor_in_workspace moe_a2a_dispatch moe_a2a_combine moe_a2a_sanitize_expert_ids moe_a2a_get_workspace_size_per_rank MoeAlltoAll",
        "属于 TensorRT-LLM MNNVL MoE all-to-all 的 workspace、payload 与 CUDA kernel 协议；HCCL 需要另行设计张量通信接口。",
    ),
    "comm.nvshmem": (
        "get_unique_id alloc_empty_unique_id init alltoall finalize my_pe n_pes malloc free_tensor barrier_all barrier_all_on_current_stream",
        "绑定 NVIDIA NVSHMEM 的 PE、symmetric heap、unique id 和 CUDA stream；昇腾 HCCL 不提供兼容的 NVSHMEM 协议。",
    ),
    "comm.nvshmem_allreduce": (
        "NVSHMEMAllReduce",
        "依赖 NVSHMEM symmetric heap 和 CUDA all-reduce kernel；不把 HCCL 的实现冒充为同一运行时。",
    ),
    "green_ctx": (
        "create_green_ctx_streams split_device_green_ctx split_device_green_ctx_by_sm_count",
        "CUDA green context 按 NVIDIA SM 切分资源；Ascend AI Core 不具有相同的 CUDA context/stream 资源契约。",
    ),
    "quantization.fp4_quantization": (
        "shuffle_matrix_a shuffle_matrix_sf_a block_scale_interleave",
        "接口固定 TensorRT-LLM/CUTLASS NVFP4 的行置换或 swizzled scale 布局；Ascend ND/NZ 不是相同的物理排布。",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="/home/dy/projects/flashinfer-ori")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    inventory = []
    for module, (names, reason) in SOURCES.items():
        path = "flashinfer/" + module.replace(".", "/") + ".py"
        source = subprocess.check_output(
            ["git", "-C", args.upstream, "show", REF + ":" + path], text=True
        )
        declarations = {n.name: n for n in ast.parse(source).body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for name in names.split():
            node = declarations[name]  # Fail if the pinned declaration is absent.
            symbol = "flashinfer." + module + "." + name
            inventory.append(dict(
                upstream=symbol, local=symbol, upstream_kind="stable",
                semantic_status="intentionally_different",
                implementation_status="missing", priority="P3", reason=reason,
                notes="硬件/供应商专属接口，不列入昇腾待实现缺口。",
                upstream_source=dict(commit=REF, path=path, line=node.lineno,
                                     sha256=hashlib.sha256(source.encode()).hexdigest()),
            ))
    for name in ("api_parity.json", "attention_api_parity.json"):
        path = Path("tools/metadata") / name
        raw = (subprocess.check_output(["git", "show", "HEAD:" + str(path)], text=True)
               if args.stage_only else path.read_text())
        value = json.loads(raw)
        for entry in value["entries"]:
            if entry["semantic_status"] == "intentionally_different":
                entry["reason"] = entry.get("reason") or entry.get("notes") or (
                    "现有账本以 is_graph_enabled 暴露后端中立图标志；上游 CUDA 命名兼容别名的实现由 T9 核对。"
                )
        if name == "api_parity.json":
            by_name = {e["upstream"]: e for e in value["entries"]}
            by_name.update({e["upstream"]: e for e in inventory})
            value["entries"] = list(by_name.values())
        data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if args.stage_only:
            blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=data, text=True).strip()
            subprocess.run(["git", "update-index", "--cacheinfo", "100644," + blob + "," + str(path)], check=True)
        else:
            path.write_text(data)
    print(json.dumps(dict(nvidia_only=len(inventory), commit=REF), indent=2))


if __name__ == "__main__":
    main()
