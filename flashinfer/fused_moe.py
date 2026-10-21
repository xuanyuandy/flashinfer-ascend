"""Fused MoE 系列对外 API（§3.1 薄门面层）。

对照固定上游 289557772，以下计算通过 NPU 原接口提供：
- ``RoutingMethodType``：top-k 路由方法枚举（纯 Python，同上游取值）；
- ``reorder_rows_for_gated_act_gemm``：trt-llm gen 权重行重排工具
  （上游即纯 torch 实现，host 侧工具，不经 dispatcher）；
- ``fused_moe``：非量化 fused MoE，dispatch 到
  cann（npu_grouped_matmul）/ reference；
- ``fp8_per_tensor_scale_moe`` / ``fp8_block_scale_moe``：
  fp8 MoE（SM100 特性），910B 无 fp8 GEMM 算子，仅注册 reference kernel
  （CPU 反量化 fp32 计算），保证注册/调度/门面链路可用（用户口径：950
  特性在 910B 只保证编译通过）。

命名说明（对标账本登记 intentionally_different）：
上游接口名中的 ``cutlass_`` / ``trtllm_`` 前缀是 NVIDIA 专用后端库
（CUTLASS / TensorRT-LLM gen），昇腾 NPU 无对应概念，故本仓主接口名
去除该前缀。这些 NPU 接口的参数顺序、选项及返回值与固定上游有差异，
因此不再暴露误称 drop-in 的三个前缀别名。
"""

from __future__ import annotations

from typing import List, Optional, Union

from .runtime import (
    Backend,
    build_signature,
    current_hardware,
    dispatch,
    get_global_registry,
)
from .runtime.schema import DispatchError

_OP_CUTLASS_FUSED_MOE = "fused_moe.cutlass_fused_moe"
_OP_TRTLLM_FP8_PER_TENSOR_MOE = "fused_moe.trtllm_fp8_per_tensor_scale_moe"
_OP_TRTLLM_FP8_BLOCK_MOE = "fused_moe.trtllm_fp8_block_scale_moe"


def _run_selected(kernel_id: str, *args, **kwargs):
    """拉起 dispatch 选中的 kernel。产物未接入时显式报错。"""
    descriptor = get_global_registry().get(kernel_id)
    if descriptor.runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入"
            "（AOT 产物加载 / JIT 编译拉起随后续里程碑接入，§15）" % kernel_id
        )
    return descriptor.runner(*args, **kwargs)


def _dtype_name(dtype) -> str:
    name = str(dtype)
    return name[len("torch."):] if name.startswith("torch.") else name


from .tllm_enums import RoutingMethodType


def _reorder_row_indices(m: int):
    """[r0..r(M-1)] → [r0, rM/2, r1, rM/2+1, ..., r(M/2-1), r(M-1)]。"""
    import torch

    row_indices = torch.arange(m, dtype=torch.long)
    top = row_indices[: (m + 1) // 2]
    bot = row_indices[(m + 1) // 2 :]
    permuted = torch.empty_like(row_indices)
    permuted[0::2] = top
    permuted[1::2] = bot
    return permuted


def reorder_rows_for_gated_act_gemm(x):
    """trt-llm gen ``reorderRowsForGatedActGemm`` 的 PyTorch 实现（对标上游）。

    将 2D 权重 (M, K) 的行上下两半交错重排，M 必须为偶数。
    """
    import torch

    if x.dim() != 2:
        raise AssertionError(f"x should be a 2D tensor, not {x.dim()}")
    m, _ = x.shape
    if m % 2 != 0:
        raise AssertionError(f"x.shape[0] must be even, not {m}")
    return x[_reorder_row_indices(m)]


def fused_moe(
    input,
    token_selected_experts,
    token_final_scales,
    fc1_expert_weights,
    fc2_expert_weights,
    output_dtype,
    quant_scales: List,
    input_sf=None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output=None,
    use_fp8_block_scaling: bool = False,
    use_w4a8_group_scaling: bool = False,
    min_latency_mode: bool = False,
    tune_max_num_tokens: int = 8192,
    *,
    moe_context=None,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """非量化 fused MoE（对标上游 cutlass_fused_moe；NPU 无 CUTLASS
    概念，主名去除后端前缀；签名差异已登记）。

    对每个 token 按 ``token_selected_experts`` 选中的专家依次执行
    ``swiglu(x @ fc1_e) @ fc2_e``，按 ``token_final_scales`` 加权聚合，
    输出 (seq_len, hidden_size)。

    当前仅支持非量化路径（input 为 fp32/fp16/bf16）；量化 scale
    （``quant_scales``/``input_sf``）在非量化路径下接受并忽略。
    fp8/nvfp4 输入、fp8 block scaling、w4a8 group scaling、
    min-latency 模式与 tp/cluster 并行暂未实现，显式报错
    （与上游 Blackwell 行为一致）。

    专家并行（``ep_size > 1``）底层走 CANN megaMoe 融合算子
    （dispatch/combine 通信内嵌），此时必须通过 ``moe_context``
    显式传入通信域对象（由
    ``cann_ops_transformer.ops.mega_moe.get_symm_buffer_for_mega_moe``
    在算子外创建），且 ``fc1/fc2_expert_weights`` 为本卡专家分片。
    单卡（默认）走 ``npu_grouped_matmul``，无需 ``moe_context``。

    Parameters
    ----------
    input : torch.Tensor
        输入 (seq_len, hidden_size)，fp32/fp16/bf16。
    token_selected_experts : torch.Tensor
        每 token 选中的专家索引 (seq_len, top_k)。
    token_final_scales : torch.Tensor
        每 token 各专家的路由权重 (seq_len, top_k)。
    fc1_expert_weights : torch.Tensor
        GEMM1 权重 (num_experts, 2*intermediate_size, hidden_size)。
    fc2_expert_weights : torch.Tensor
        GEMM2 权重 (num_experts, hidden_size, intermediate_size)。
    output_dtype : torch.dtype
        输出 dtype。
    quant_scales / input_sf
        量化参数，非量化路径下忽略。
    tp_size / tp_rank / ep_size / ep_rank / cluster_size / cluster_rank
        并行参数；tp/cluster 当前仅支持 1。``ep_size > 1`` 时启用
        专家并行（megaMoe 路径）。
    moe_context
        EP 通信域对象（duck-typed，需含 ``context``/``ep_world_size``/
        ``ccl_buffer_size`` 等属性，由
        ``cann_ops_transformer.ops.mega_moe.get_symm_buffer_for_mega_moe``
        创建）。``ep_size > 1`` 时必传；``ep_size == 1``（单卡）时忽略，
        传与不传行为一致。与 ``backend``/``kernel_id`` 同属语义正交的
        资源参数，不参与调度签名。
    output : Optional[torch.Tensor]
        输出张量，缺省时分配。
    tune_max_num_tokens : int
        调优参数，NPU 无对应机制，接受并忽略。
    backend / kernel_id
        调度覆盖参数（§3.2.2，与计算语义正交）。

    Returns
    -------
    output : torch.Tensor
        shape (seq_len, hidden_size)。
    """
    if use_fp8_block_scaling:
        raise NotImplementedError("FP8 Block Scaling 暂未实现（同上游 Blackwell）")
    if use_w4a8_group_scaling:
        raise NotImplementedError("W4A8 Group Scaling 暂未实现（同上游 Blackwell）")
    if min_latency_mode:
        raise NotImplementedError("min latency 模式暂未实现（同上游 Blackwell）")
    if tp_size != 1 or cluster_size != 1:
        raise NotImplementedError("v1 仅支持单卡 tp/cluster（均为 1）")
    if ep_size < 1:
        raise ValueError(f"ep_size 须 >= 1，got {ep_size}")
    if ep_size > 1:
        if moe_context is None:
            raise ValueError(
                "ep_size > 1 时必传 moe_context（由 cann_ops_transformer.ops."
                "mega_moe.get_symm_buffer_for_mega_moe 创建）"
            )
        ctx_ep = getattr(moe_context, "ep_world_size", None)
        if ctx_ep is not None and ctx_ep != ep_size:
            raise ValueError(
                f"moe_context.ep_world_size({ctx_ep}) 与 ep_size({ep_size}) 不一致"
            )
        ctx_rank = getattr(moe_context, "rank_id", None)
        if ctx_rank is not None and ctx_rank != ep_rank:
            raise ValueError(
                f"moe_context.rank_id({ctx_rank}) 与 ep_rank({ep_rank}) 不一致"
            )
    if _dtype_name(input.dtype) in ("float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError("fp8 输入请走 fp8_per_tensor/block_scale_moe 接口")
    del tp_rank, cluster_rank, quant_scales, input_sf
    del tune_max_num_tokens

    import torch  # 延迟导入：模块本身不依赖 torch（§14.1 host 单测）

    num_rows = input.shape[0]
    hidden_size = fc2_expert_weights.shape[1]
    if output is None:
        output = torch.empty(
            num_rows, hidden_size, dtype=output_dtype, device=input.device
        )
    signature = build_signature(
        _OP_CUTLASS_FUSED_MOE,
        (
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
        ),
        specializations={"output_dtype": _dtype_name(output_dtype), "ep_size": ep_size},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(
        decision.kernel_id,
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        output_dtype,
        output,
        ep_size,
        ep_rank,
        moe_context,
    )
    return output


def fp8_per_tensor_scale_moe(
    routing_logits,
    routing_bias,
    hidden_states,
    gemm1_weights,
    output1_scales_scalar,
    output1_scales_gate_scalar,
    gemm2_weights,
    output2_scales_scalar,
    output,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: float,
    use_routing_scales_on_input: bool,
    tile_tokens_dim: int = 8,
    routing_method_type: int = 0,
    store_workspace_info: bool = False,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
) -> None:
    """FP8 per-tensor scale MoE（对标上游 trtllm_fp8_per_tensor_scale_moe；
    NPU 无 TRT-LLM 概念，主名去除后端前缀；签名差异已登记）。

    路由（``routing_method_type``，见 :class:`RoutingMethodType`）→
    选中专家执行 fp8 双 GEMM（per-tensor scale 反量化）→ 结果原地写入
    ``output``，返回 None。910B 无 fp8 GEMM 算子，仅 reference kernel
    可执行（反量化 fp32 计算）。``tile_tokens_dim`` 为 SM100 调度参数，
    接受并忽略。
    """
    signature = build_signature(
        _OP_TRTLLM_FP8_PER_TENSOR_MOE,
        (routing_logits, hidden_states, gemm1_weights, gemm2_weights),
        specializations={
            "top_k": int(top_k),
            "routing_method_type": int(routing_method_type),
        },
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(
        decision.kernel_id,
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights,
        output2_scales_scalar,
        output,
        int(num_experts),
        int(top_k),
        int(n_group),
        int(topk_group),
        int(intermediate_size),
        int(local_expert_offset),
        int(local_num_experts),
        float(routed_scaling_factor),
        bool(use_routing_scales_on_input),
        int(tile_tokens_dim),
        int(routing_method_type),
        bool(store_workspace_info),
    )
    return None


def fp8_block_scale_moe(
    expert_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_scales,
    gemm2_weights,
    gemm2_scales,
    output,
    num_experts: int,
    top_k: int,
    n_groups: int,
    top_k_groups: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling: float,
    tile_tokens_dim: int = 8,
    routing_method_type: int = 0,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
) -> None:
    """FP8 block scale MoE（对标上游 trtllm_fp8_block_scale_moe；
    NPU 无 TRT-LLM 概念，主名去除后端前缀；签名差异已登记）。

    与 per-tensor 版的差异在量化粒度：hidden/gemm 权重均按 128-block
    携带 scale（``hidden_states_scale`` 布局 (hidden//128, seq_len)）。
    结果原地写入 ``output``，返回 None。910B 仅 reference kernel 可执行。
    """
    signature = build_signature(
        _OP_TRTLLM_FP8_BLOCK_MOE,
        (expert_logits, hidden_states, gemm1_weights, gemm2_weights),
        specializations={
            "top_k": int(top_k),
            "routing_method_type": int(routing_method_type),
        },
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(
        decision.kernel_id,
        expert_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_scales,
        gemm2_weights,
        gemm2_scales,
        output,
        int(num_experts),
        int(top_k),
        int(n_groups),
        int(top_k_groups),
        int(intermediate_size),
        int(local_expert_offset),
        int(local_num_experts),
        float(routed_scaling),
        int(tile_tokens_dim),
        int(routing_method_type),
    )
    return None
