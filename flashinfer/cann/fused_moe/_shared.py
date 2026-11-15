"""fused_moe 系列：CANN 内置算子薄封装（§6.4）。

对齐上游 ``flashinfer.fused_moe.cutlass_fused_moe``（非量化路径）：
路由结果（token_selected_experts / token_final_scales）由调用方给定，
本封装负责专家分组与两段 GEMM——底层走 CANN 已有算子
``npu_grouped_matmul``（非量化公式：y_i = x_i × weight_i）。

流程：按 expert 排序聚簇 token → fc1 grouped GEMM → swiglu →
fc2 grouped GEMM → 反置换 + 路由权重聚合（index_add 语义）。

CANN 版本适配说明：本环境（910B, CANN 8.x）上 group_list 需以
python list 形式传入（tensor 形式触发 aclnnGroupedMatmulV5 161002），
返回值为按组拆分的 tensor list，空组返回 (0, n)。
"""

from __future__ import annotations

from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

_SUPPORTED_DTYPES = ("float16", "bfloat16", "int32", "int64", "float32")

_SUPPORTED_SOCS = GENERAL_ASCEND_SOCS

_FP8_SUPPORTED_DTYPES = _SUPPORTED_DTYPES + ("float8_e4m3fn",)

_FP8_SUPPORTED_SOCS = ("Ascend950",)

def _fused_moe_megamoe(
    input: Any,
    token_selected_experts: Any,
    token_final_scales: Any,
    fc1_expert_weights: Any,
    fc2_expert_weights: Any,
    output: Any,
    moe_context: Any,
) -> Any:
    """EP 分支：底层走 CANN megaMoe 融合算子（dispatch/combine 通信内嵌）。

    ``moe_context`` 为 cann_ops_transformer 的 SymmBuffer（通信域对象，
    算子外创建）；fc1/fc2 为本卡专家分片，topk_ids 为全局专家索引，
    fc1 前半 gate、后半 up。

    布局转换：flashinfer 接口沿用上游约定 fc1 (E, 2I, H) / fc2 (E, H, I)，
    而 megaMoe 的 weight1/weight2 要求每专家 (H, 2I) / (I, H)（K 在 dim0），
    此处按专家转置后传入（megaMoe 另要求 inter >= 512 且 %512==0）。
    """
    try:
        from cann_ops_transformer.ops.mega_moe import mega_moe as _npu_mega_moe
    except ImportError as exc:
        raise RuntimeError(
            "ep_size > 1 需安装 cann_ops_transformer（含 mega_moe 算子），"
            "按 ops-transformer 仓 readme 编译安装后重试"
        ) from exc

    w1_list = [w.t().contiguous() for w in fc1_expert_weights.unbind(0)]
    w2_list = [w.t().contiguous() for w in fc2_expert_weights.unbind(0)]
    y, _expert_token_nums = _npu_mega_moe(
        input,
        token_selected_experts.int(),
        token_final_scales,
        w1_list,
        w2_list,
        moe_context,
        activation="swiglu",
    )
    output.copy_(y.to(output.dtype))
    return output

def _grouped_matmul(x: Any, weight_list: list, group_list: list) -> Any:
    """单 tensor 输入 + cumsum 分组的 grouped GEMM，返回拼接后的 (N, n)。"""
    import torch
    import torch_npu

    y_list = torch_npu.npu_grouped_matmul(
        [x],
        weight_list,
        group_list=group_list,
        split_item=1,
        group_type=0,
    )
    return torch.cat(list(y_list), dim=0)

def _cluster_tokens(
    input: Any, token_selected_experts: Any, num_experts: int
) -> Any:
    """token 按专家聚簇：每 token 展开 top_k 行后按专家 id 稳定排序。

    返回 (x_sorted, inverse, group_list, e_of_row, order)：x_sorted 为按
    专家连续排列的 (S*topk, H)；inverse 为反置换；group_list 为各专家
    行数前缀和（grouped GEMM 入参）；e_of_row 为排序后每行的专家 id；
    order 为排序置换（供调用方同步排列其它按行对齐的张量）。
    """
    import torch

    top_k = token_selected_experts.shape[1]
    flat_e = token_selected_experts.reshape(-1).long()  # (S*topk,)
    # FP8 的 repeat_interleave / index 尚未实现；等宽字节视图只搬运
    # 位模式，不反量化、不经过 bf16，GEMM 输入仍为原始 FP8。
    is_fp8 = input.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    storage = input.view(torch.uint8) if is_fp8 else input
    x_rep = storage.repeat_interleave(top_k, dim=0)  # (S*topk, H)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e.cpu(), minlength=num_experts)
    group_list = counts.cumsum(0).tolist()  # 非量化兼容路径保留 list；FP8 内部物化 Tensor
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    x_sorted = x_rep[order].contiguous()
    if is_fp8:
        x_sorted = x_sorted.view(input.dtype)
    return x_sorted, inverse, group_list, flat_e[order], order

def _aggregate_expert_output(
    y2: Any,
    inverse: Any,
    token_final_scales: Any,
    seq: int,
    top_k: int,
    hidden: int,
) -> Any:
    """反置换回 (S*topk, H)，乘路由权重后按 token 聚合为 (S, H)。"""
    y2 = y2[inverse] * token_final_scales.reshape(-1, 1).to(y2.dtype)
    return y2.reshape(seq, top_k, hidden).sum(dim=1)

def _gmm_expert_compute(
    input: Any,
    token_selected_experts: Any,
    token_final_scales: Any,
    fc1_expert_weights: Any,
    fc2_expert_weights: Any,
) -> Any:
    """单卡 MoE 计算链（小算子级联）：token 按专家聚簇 → fc1 grouped
    GEMM → swiglu → fc2 grouped GEMM → 反置换 + 路由权重聚合。

    返回 (S, H) 聚合结果（未写入 output）；input/权重需为 GEMM 支持的
    dtype（fp16/bf16/fp32），反量化等预处理由调用方完成。
    """
    import torch

    seq, hidden = input.shape
    num_experts = fc1_expert_weights.shape[0]
    inter = fc1_expert_weights.shape[1] // 2
    top_k = token_selected_experts.shape[1]

    x_sorted, inverse, group_list, _, _ = _cluster_tokens(
        input, token_selected_experts, num_experts
    )
    # fc1：x_sorted @ W1_e^T（weight 为 (k, n) = (H, 2I)，转置物化）
    w1_list = [fc1_expert_weights[e].t().contiguous() for e in range(num_experts)]
    y1 = _grouped_matmul(x_sorted, w1_list, group_list)  # (S*topk, 2I)
    h = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    # fc2：h @ W2_e^T（(k, n) = (I, H)）
    w2_list = [fc2_expert_weights[e].t().contiguous() for e in range(num_experts)]
    y2 = _grouped_matmul(h.contiguous(), w2_list, group_list)  # (S*topk, H)
    return _aggregate_expert_output(
        y2, inverse, token_final_scales, seq, top_k, hidden
    )

def _mask_nonlocal_experts(idx: Any, w: Any, offset: int, num_local: int) -> Any:
    """EP 语义：区间 [offset, offset+num_local) 外专家的路由权重置零。"""
    if offset == 0 and num_local >= int(idx.max()) + 1:
        return w
    in_range = (idx >= offset) & (idx < offset + num_local)
    return w * in_range.to(w.dtype)

_FP8_E4M3_MAX = 448.0

def _fp8_gmm(x8: Any, weight_list_t: list, group_list: list) -> Any:
    """fp8 e4m3 grouped GEMM（950 上 ACLNN grouped matmul 的 fp8 路径）。

    x8 为按专家聚簇排列的 (rows, k) fp8；weight_list_t 为逐专家 (k, n)
    转置权重（fp8）；输出按专家拼接的 (rows, n)。

    与 GEMM 共用 Tensor group_list、单位 scale 和 fp32 输出的 ACLNN
    FP8 路径，保持外部 scale 折算前的点积精度与动态范围。
    """
    from flashinfer.cann.gemm import _fp8_grouped_gemm

    return _fp8_grouped_gemm(x8, weight_list_t, group_list)

def _fp8_quant_per_row(h: Any) -> Any:
    """逐行动态量化到 e4m3fn，返回 (h8, scale (rows, 1) fp32)。"""
    import torch

    amax = h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / _FP8_E4M3_MAX).float()
    h8 = (h / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return h8.to(torch.float8_e4m3fn), scale

def _fp8_quant_per_row_block(h: Any, block: int = 128) -> Any:
    """逐行 128-block 动态量化（DeepSeek 形态），返回
    (h8, scale (rows, n//block) fp32)。"""
    import torch

    rows, n = h.shape
    hb = h.reshape(rows, n // block, block)
    amax = hb.abs().amax(dim=-1).clamp(min=1e-12)
    scale = (amax / _FP8_E4M3_MAX).float()
    h8 = (hb / scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return h8.reshape(rows, n).to(torch.float8_e4m3fn), scale

def _fp8_block_gmm(
    x8: Any,
    xs: Any,
    w8: Any,
    ws: Any,
    group_list: list,
    e_of_row: Any,
) -> Any:
    """128×128 block-scale fp8 grouped GEMM：K 方向按 128 分块累加。

    x8 (rows, K) fp8，按专家聚簇排列；xs (rows, K//128) fp32（行 = 排序
    后 token 行，行方向不分块）；w8 (E, N, K) fp8；ws (E, N//128, K//128)
    fp32。返回 fp32 (rows, N)（block scale 折算后累加结果）。
    """
    import torch

    num_experts, n_dim, k_dim = w8.shape
    y = torch.zeros(
        x8.shape[0], n_dim, dtype=torch.float32, device=x8.device
    )
    for c in range(k_dim // 128):
        sl = slice(c * 128, (c + 1) * 128)
        w_list_t = [w8[e][:, sl].t().contiguous() for e in range(num_experts)]
        part = _fp8_gmm(x8[:, sl].contiguous(), w_list_t, group_list)
        # 行 scale：xs[:, c]；列 block scale：ws[e, n_block, c] 按行专家 gather
        scale = xs[:, c : c + 1] * ws[e_of_row, :, c]  # (rows, N//128)
        y += part.float() * scale.repeat_interleave(128, dim=1)
    return y
