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

from typing import Any, Tuple

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# input/权重 fp16/bf16；expert 索引 int32/int64、路由权重 fp32 一并列入
# （capability 检查覆盖签名全部 dtype）
_SUPPORTED_DTYPES = ("float16", "bfloat16", "int32", "int64", "float32")
# 单卡分支（npu_grouped_matmul）与 EP 分支（megaMoe，ops-transformer 注册
# 覆盖 910b/910_93/950）共用同一 descriptor
_SUPPORTED_SOCS = ("Ascend910B", "Ascend910_93", "Ascend950")
# fp8 接口：激活/权重 e4m3fn + 路由/scale fp32 + 输出 bf16/fp16
_FP8_SUPPORTED_DTYPES = _SUPPORTED_DTYPES + ("float8_e4m3fn",)
# fp8 计算仅 950 支持（910B 无 fp8 硬件单元）
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
    x_rep = input.repeat_interleave(top_k, dim=0)  # (S*topk, H)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e.cpu(), minlength=num_experts)
    group_list = counts.cumsum(0).tolist()  # CANN 本版本要求 python list
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return x_rep[order].contiguous(), inverse, group_list, flat_e[order], order


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


def _cutlass_fused_moe_gmm(
    input: Any,
    token_selected_experts: Any,
    token_final_scales: Any,
    fc1_expert_weights: Any,
    fc2_expert_weights: Any,
    output_dtype: Any,
    output: Any,
    ep_size: int = 1,
    ep_rank: int = 0,
    moe_context: Any = None,
) -> Any:
    if ep_size > 1:
        del output_dtype  # 输出精度以 output 张量 dtype 为准
        return _fused_moe_megamoe(
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            output,
            moe_context,
        )
    del ep_rank

    out = _gmm_expert_compute(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
    )
    del output_dtype  # 输出精度以 output 张量 dtype 为准
    output.copy_(out.to(output.dtype))
    return output


_FP8_E4M3_MAX = 448.0


def _fp8_gmm(x8: Any, weight_list_t: list, group_list: list) -> Any:
    """fp8 e4m3 grouped GEMM（950 上 ACLNN grouped matmul 的 fp8 路径）。

    x8 为按专家聚簇排列的 (rows, k) fp8；weight_list_t 为逐专家 (k, n)
    转置权重（fp8）；输出按专家拼接的 (rows, n)。

    950 验证点：若本 CANN 版本的 npu_grouped_matmul 不直接接受 fp8 输入
    （或输出 dtype 推断不符合预期），需切换为对应 aclnn fp8 形态。
    """
    import torch
    import torch_npu

    y_list = torch_npu.npu_grouped_matmul(
        [x8],
        weight_list_t,
        group_list=group_list,
        split_item=1,
        group_type=0,
    )
    return torch.cat(list(y_list), dim=0)


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


def _fp8_per_tensor_scale_moe_cascade(
    routing_logits: Any,
    routing_bias: Any,
    hidden_states: Any,
    gemm1_weights: Any,
    output1_scales_scalar: Any,
    output1_scales_gate_scalar: Any,
    gemm2_weights: Any,
    output2_scales_scalar: Any,
    output: Any,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    use_routing_scales_on_input: bool,
    tile_tokens_dim: int,
    routing_method_type: int,
    store_workspace_info: bool,
) -> None:
    """fp8_per_tensor_scale_moe CANN 路径（950）：纯 fp8 小算子级联。

    数据流：路由（fp32 小算子）→ token 聚簇 → fc1 fp8 grouped GEMM →
    逐专家 scale（s1×s1_gate）折算 → swiglu → 逐行动态量化回 fp8 →
    fc2 fp8 grouped GEMM → scale（s2×中间量化系数）折算 → 反置换 +
    路由权重聚合。GEMM 全程 fp8 计算，激活/权重无 bf16 中转、无重量化。
    use_routing_scales_on_input 在线性场景数学等价，按输出侧处理。
    """
    import torch

    from ..runtime.reference_kernels import _route_topk

    del intermediate_size, use_routing_scales_on_input, tile_tokens_dim
    del store_workspace_info
    idx, w = _route_topk(
        routing_logits.float(),
        routing_bias.float(),
        top_k,
        n_group,
        topk_group,
        routed_scaling_factor,
        routing_method_type,
    )
    w = _mask_nonlocal_experts(idx, w, local_expert_offset, local_num_experts)
    seq, hidden = hidden_states.shape
    inter = gemm1_weights.shape[1] // 2

    x_sorted, inverse, group_list, e_of_row, _ = _cluster_tokens(
        hidden_states, idx, num_experts
    )
    # fc1：fp8 GEMM，per-tensor scale 按行所属专家折算到输出
    w1_t = [gemm1_weights[e].t().contiguous() for e in range(num_experts)]
    y1 = _fp8_gmm(x_sorted, w1_t, group_list).float()
    c1 = output1_scales_scalar.float() * output1_scales_gate_scalar.float()
    y1 *= c1[e_of_row].unsqueeze(-1)
    # swiglu → 逐行动态量化回 fp8 供 fc2
    h = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    h8, hs = _fp8_quant_per_row(h)
    # fc2：fp8 GEMM，scale = s2 × 中间量化系数
    w2_t = [gemm2_weights[e].t().contiguous() for e in range(num_experts)]
    y2 = _fp8_gmm(h8.contiguous(), w2_t, group_list).float()
    y2 *= (output2_scales_scalar.float()[e_of_row] * hs.squeeze(-1)).unsqueeze(-1)
    out = _aggregate_expert_output(y2, inverse, w, seq, top_k, hidden)
    output.copy_(out.to(output.dtype))
    return None


def _fp8_block_scale_moe_cascade(
    expert_logits: Any,
    routing_bias: Any,
    hidden_states: Any,
    hidden_states_scale: Any,
    gemm1_weights: Any,
    gemm1_scales: Any,
    gemm2_weights: Any,
    gemm2_scales: Any,
    output: Any,
    num_experts: int,
    top_k: int,
    n_groups: int,
    top_k_groups: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling: float,
    tile_tokens_dim: int,
    routing_method_type: int,
) -> None:
    """fp8_block_scale_moe CANN 路径（950）：纯 fp8 小算子级联。

    激活 scale 布局 (H//128, S)（上游转置存放），权重 scale
    (E, N//128, K//128)；GEMM 按 K 方向 128 分块调用 fp8 grouped GEMM
    并以 block scale 折算累加；swiglu 输出按逐行 128-block 动态量化回
    fp8 供 fc2。全程 fp8 计算，无 bf16 中转、无重量化。
    """
    import torch

    from ..runtime.reference_kernels import _route_topk

    del intermediate_size, tile_tokens_dim
    idx, w = _route_topk(
        expert_logits.float(),
        routing_bias.float(),
        top_k,
        n_groups,
        top_k_groups,
        routed_scaling,
        routing_method_type,
    )
    w = _mask_nonlocal_experts(idx, w, local_expert_offset, local_num_experts)
    seq, hidden = hidden_states.shape
    inter = gemm1_weights.shape[1] // 2

    x_sorted, inverse, group_list, e_of_row, order = _cluster_tokens(
        hidden_states, idx, num_experts
    )
    # 激活 block scale 与聚簇后的行对齐（行方向不分块）
    xs = hidden_states_scale.float().t()  # (S, H//128)
    xs_sorted = xs.repeat_interleave(top_k, dim=0)[order].contiguous()
    y1 = _fp8_block_gmm(
        x_sorted, xs_sorted, gemm1_weights, gemm1_scales.float(),
        group_list, e_of_row,
    )
    h = torch.nn.functional.silu(y1[:, :inter]) * y1[:, inter:]
    h8, hs = _fp8_quant_per_row_block(h)
    y2 = _fp8_block_gmm(
        h8.contiguous(), hs, gemm2_weights, gemm2_scales.float(),
        group_list, e_of_row,
    )
    out = _aggregate_expert_output(y2, inverse, w, seq, top_k, hidden)
    output.copy_(out.to(output.dtype))
    return None


def moe_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN fused_moe 注册表。"""
    return (
        KernelDescriptor(
            kernel_id="cann/fused_moe/cutlass_fused_moe_gmm_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="fused_moe.cutlass_fused_moe",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_cutlass_fused_moe_gmm,
        ),
        KernelDescriptor(
            kernel_id="cann/fused_moe/fp8_per_tensor_scale_moe_cascade_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="fused_moe.trtllm_fp8_per_tensor_scale_moe",
                dtypes=_FP8_SUPPORTED_DTYPES,
                soc_models=_FP8_SUPPORTED_SOCS,
            ),
            runner=_fp8_per_tensor_scale_moe_cascade,
        ),
        KernelDescriptor(
            kernel_id="cann/fused_moe/fp8_block_scale_moe_cascade_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="fused_moe.trtllm_fp8_block_scale_moe",
                dtypes=_FP8_SUPPORTED_DTYPES,
                soc_models=_FP8_SUPPORTED_SOCS,
            ),
            runner=_fp8_block_scale_moe_cascade,
        ),
    )
