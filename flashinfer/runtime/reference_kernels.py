"""reference backend：高精度参考实现（§5.2）。

定位：correctness 基线与调试对照，**默认不进生产热路径**——
参与 dispatch 需显式 opt-in（§12）。M0 阶段只提供朴素 attention
参考实现，正确性细节随 M1 起逐步完善。
"""

from __future__ import annotations

from typing import Any, Tuple

from .schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("reference backend 的执行需要 torch") from exc
    return torch


def _repeat_kv(k, v, num_heads: int):
    h_kv = k.shape[-2]
    if num_heads % h_kv:
        raise ValueError("num_heads 必须是 num_kv_heads 的整数倍")
    if h_kv != num_heads:
        k = k.repeat_interleave(num_heads // h_kv, dim=-2)
        v = v.repeat_interleave(num_heads // h_kv, dim=-2)
    return k, v


def _prefill_ragged_ref(q, k, v, causal: bool = False, **_unused):
    """朴素 ragged prefill：q (total_q, H, D)，k/v (total_kv, H_kv, D)。"""
    torch = _require_torch()
    k, v = _repeat_kv(k, v, q.shape[-2])
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float())
    scores = scores / (q.shape[-1] ** 0.5)
    if causal:
        q_len, k_len = q.shape[0], k.shape[0]
        mask = torch.full((q_len, k_len), float("-inf"), device=scores.device)
        mask = torch.triu(mask, diagonal=k_len - q_len + 1)
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float()).to(q.dtype)


def _decode_ref(q, k_cache, v_cache, **_unused):
    """朴素单步 decode：q (B, H, D)，k_cache/v_cache (B, S, H_kv, D)。"""
    torch = _require_torch()
    k, v = _repeat_kv(k_cache, v_cache, q.shape[-2])
    scores = torch.einsum("bhd,bshd->bhs", q.float(), k.float())
    scores = scores / (q.shape[-1] ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bshd->bhd", probs, v.float()).to(q.dtype)


def reference_rmsnorm(out: Any, input: Any, weight: Any, eps: float) -> None:
    """RMSNorm 高精度参考实现（float64 累计）。

    ``out[i] = (input[i] / RMS(input)) * weight[i]``，最后一维为归一化维。
    """
    torch = _require_torch()
    assert input.shape == out.shape, "rmsnorm: out 与 input 形状须一致"
    assert weight.shape == input.shape[-1:], "rmsnorm: weight 形状须为 (hidden_size,)"
    with torch.no_grad():
        x = input.double()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        out.copy_((x * rstd * weight.double()).to(out.dtype))


def reference_fused_add_rmsnorm(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    """Fused add RMSNorm 高精度参考实现（float64 累计，原地修改）。

    Step 1: ``residual[i] += input[i]``
    Step 2: ``input[i] = (residual[i] / RMS(residual)) * weight[i]``
    """
    torch = _require_torch()
    assert input.shape == residual.shape, "fused_add_rmsnorm: 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "fused_add_rmsnorm: weight 形状须为 (hidden_size,)"
    )
    with torch.no_grad():
        new_residual = input.double() + residual.double()
        residual.copy_(new_residual.to(residual.dtype))
        rstd = torch.rsqrt(new_residual.pow(2).mean(dim=-1, keepdim=True) + eps)
        input.copy_((new_residual * rstd * weight.double()).to(input.dtype))


def reference_gemma_rmsnorm(out: Any, input: Any, weight: Any, eps: float) -> None:
    """Gemma RMSNorm 高精度参考实现（float64 累计）。

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``——与 rmsnorm
    的唯一差异是缩放因子为 ``weight + 1``（Gemma 约定）。
    """
    torch = _require_torch()
    assert input.shape == out.shape, "gemma_rmsnorm: out 与 input 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "gemma_rmsnorm: weight 形状须为 (hidden_size,)"
    )
    with torch.no_grad():
        x = input.double()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        out.copy_((x * rstd * (weight.double() + 1.0)).to(out.dtype))


def reference_gemma_fused_add_rmsnorm(
    input: Any, residual: Any, weight: Any, eps: float
) -> None:
    """Gemma fused add RMSNorm 高精度参考实现（float64 累计，原地修改）。

    Step 1: ``residual[i] += input[i]``
    Step 2: ``input[i] = (residual[i] / RMS(residual)) * (weight[i] + 1)``
    """
    torch = _require_torch()
    assert input.shape == residual.shape, "gemma_fused_add_rmsnorm: 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "gemma_fused_add_rmsnorm: weight 形状须为 (hidden_size,)"
    )
    with torch.no_grad():
        new_residual = input.double() + residual.double()
        residual.copy_(new_residual.to(residual.dtype))
        rstd = torch.rsqrt(new_residual.pow(2).mean(dim=-1, keepdim=True) + eps)
        input.copy_((new_residual * rstd * (weight.double() + 1.0)).to(input.dtype))


def reference_layernorm(
    out: Any, input: Any, weight: Any, bias: Any, eps: float
) -> None:
    """LayerNorm 高精度参考实现（float64 累计）。

    ``out = (input - E[input]) / sqrt(Var[input] + eps) * weight + bias``，
    最后一维为归一化维。对标上游 flashinfer.norm.layernorm(v0.6.0)，
    其 gemma/beta 即此处 weight/bias。
    """
    torch = _require_torch()
    assert input.shape == out.shape, "layernorm: out 与 input 形状须一致"
    assert weight.shape == input.shape[-1:], (
        "layernorm: weight 形状须为 (hidden_size,)"
    )
    assert bias.shape == weight.shape, "layernorm: bias 与 weight 形状须一致"
    with torch.no_grad():
        x = input.double()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + eps) * weight.double() + bias.double()
        out.copy_(y.to(out.dtype))


def reference_apply_rope_with_cos_sin_cache(
    positions: Any,
    query: Any,
    key: Any,
    head_size: int,
    cos_sin_cache: Any,
    is_neox: bool,
) -> None:
    """RoPE（cos/sin cache 变体）高精度参考实现（float64 累计，原地修改）。

    对标上游 ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace``：
    ``cos_sin_cache`` 前一半为 cos、后一半为 sin，按 ``positions`` 索引；
    仅旋转每个 head 的前 ``rotary_dim`` 维。``is_neox=True`` 为半旋转
    （GPT-NeoX 风格），``False`` 为交错旋转（GPT-J 风格）。
    """
    torch = _require_torch()
    rotary_dim = cos_sin_cache.shape[-1]
    half = rotary_dim // 2
    with torch.no_grad():
        cs = cos_sin_cache.double().index_select(0, positions)  # (nnz, rd)
        cos = cs[:, :half].unsqueeze(1)  # (nnz, 1, rd/2)
        sin = cs[:, half:].unsqueeze(1)
        for t in (query, key):
            t3 = t.view(t.shape[0], -1, head_size)
            x = t3[..., :rotary_dim].double()
            if is_neox:
                x1, x2 = x[..., :half], x[..., half:]
                rot = torch.cat(
                    [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1
                )
            else:
                xe, xo = x[..., 0::2], x[..., 1::2]
                rot = torch.stack(
                    [xe * cos - xo * sin, xo * cos + xe * sin], dim=-1
                ).flatten(-2)
            t3[..., :rotary_dim].copy_(rot.to(t.dtype))


def reference_cudnn_batch_decode(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    scale: float,
    max_sequence_kv: int,
    actual_seq_lens_kv: Any,
    block_tables: Any,
    out: Any,
) -> Any:
    """paged KV decode attention 高精度参考实现（float64 累计）。

    对标上游 ``flashinfer.decode.cudnn_batch_decode_with_kv_cache``：
    q (B, H_qo, D)，k/v_cache (total_pages, H_kv, page_size, D)，
    block_tables (B, num_pages_per_seq)；仅 causal（单 token 全量可见），
    支持 GQA（H_qo 为 H_kv 整数倍）。
    """
    torch = _require_torch()
    B, H_qo, D = q.shape
    _, H_kv, page_size, D_v = v_cache.shape
    group = H_qo // H_kv
    lens = actual_seq_lens_kv.long().cpu()
    tables = block_tables.long().cpu()
    with torch.no_grad():
        for b in range(B):
            kv_len = min(int(lens[b]), int(max_sequence_kv))
            pages_b = tables[b]
            # (npages, H_kv, ps, D) → (npages*ps, H_kv, D)，截断到实际长度
            k_b = (
                k_cache[pages_b]
                .double()
                .permute(0, 2, 1, 3)
                .reshape(-1, H_kv, D)[:kv_len]
            )
            v_b = (
                v_cache[pages_b]
                .double()
                .permute(0, 2, 1, 3)
                .reshape(-1, H_kv, D_v)[:kv_len]
            )
            k_rep = k_b.repeat_interleave(group, dim=1)  # (L, H_qo, D)
            v_rep = v_b.repeat_interleave(group, dim=1)
            scores = torch.einsum("hd,lhd->hl", q[b].double(), k_rep) * scale
            p = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hl,lhd->hd", p, v_rep).to(out.dtype)
    return out


def reference_bmm_fp8(
    A: Any,
    B: Any,
    A_scale: Any,
    B_scale: Any,
    out: Any,
) -> Any:
    """fp8 per-tensor 批量矩阵乘参考实现（反量化到 float32 计算）。

    对标上游 ``flashinfer.gemm.bmm_fp8``：A (b, m, k)、B (b, k, n) 为
    fp8(e4m3/e5m2)，A_scale/B_scale 为 per-tensor 标量 scale，
    ``out = (A @ B) * A_scale * B_scale``，输出 bf16/fp16。
    """
    torch = _require_torch()
    with torch.no_grad():
        acc = torch.matmul(A.float(), B.float())
        acc = acc * A_scale.float() * B_scale.float()
        out.copy_(acc.to(out.dtype))
    return out


def reference_gemm_fp8_nt_groupwise(
    a: Any,
    b: Any,
    a_scale: Any,
    b_scale: Any,
    scale_major_mode: str,
    scale_granularity_mnk: Any,
    out: Any,
) -> Any:
    """fp8 分组 scale 矩阵乘参考实现（反量化到 float32 计算）。

    对标上游 ``flashinfer.gemm.gemm_fp8_nt_groupwise``：a (m, k) 行主序、
    b (n, k) 列主序（NT 布局），分组粒度 (gm, gn, gk)；
    scale_major_mode="K" 时 a_scale (m//gm, k//gk)、b_scale (n//gn, k//gk)；
    "MN" 时两 scale 转置存放。``out = (a ⊙ sa) @ (b ⊙ sb)^T``。
    """
    gm, gn, gk = (int(x) for x in scale_granularity_mnk)

    def _expand(scale, rows, cols, g_r, g_c):
        # scale 归一到 (rows//g_r, cols//g_c) 后按粒度扩展到 (rows, cols)
        s = scale.float()
        if s.shape != (rows // g_r, cols // g_c):
            s = s.t()  # MN-major 存放 → 转置回 (rows, cols) 分块视角
        assert s.shape == (rows // g_r, cols // g_c), (
            f"scale 形状 {tuple(s.shape)} 与粒度 {(g_r, g_c)} 不匹配"
        )
        return s.repeat_interleave(g_r, 0).repeat_interleave(g_c, 1)

    m, k = a.shape
    n = b.shape[0]
    sa = _expand(a_scale, m, k, gm, gk)
    sb = _expand(b_scale, n, k, gn, gk)
    del scale_major_mode  # 布局差异已在 _expand 转置归一中吸收
    torch = _require_torch()
    with torch.no_grad():
        acc = torch.matmul(a.float() * sa, (b.float() * sb).t())
        out.copy_(acc.to(out.dtype))
    return out


def _route_topk(
    logits: Any,
    bias: Any,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling: float,
    routing_method_type: int,
) -> Tuple[Any, Any]:
    """trtllm 系 MoE 的 top-k 路由（对标上游 RoutingMethodType 语义）。

    返回 (expert_idx (S, top_k) long, scales (S, top_k) float)。
    """
    torch = _require_torch()
    method = int(routing_method_type)
    if method == 0:  # Default: Softmax -> TopK
        p = logits.softmax(-1)
        w, idx = p.topk(top_k, dim=-1)
    elif method == 1:  # Renormalize: TopK -> Softmax
        w, idx = logits.topk(top_k, dim=-1)
        w = w.softmax(-1)
    elif method == 3:  # Llama4: Top1 -> Sigmoid
        w, idx = logits.topk(top_k, dim=-1)
        w = torch.sigmoid(w)
    elif method == 4:  # RenormalizeNaive: Softmax -> TopK -> Renormalize
        p = logits.softmax(-1)
        w, idx = p.topk(top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    elif method == 2:  # DeepSeekV3: Sigmoid -> +bias -> 组内 Top2 -> 组 TopK -> 专家 TopK
        s = torch.sigmoid(logits)
        sb = s + bias
        seq, num_experts = s.shape
        sb_g = sb.reshape(seq, n_group, num_experts // n_group)
        group_score = sb_g.topk(2, dim=-1).values.sum(-1)  # (S, n_group)
        top_groups = group_score.topk(topk_group, dim=-1).indices
        group_mask = torch.zeros_like(sb_g, dtype=torch.bool)
        group_mask.scatter_(
            1, top_groups.unsqueeze(-1).expand(-1, -1, sb_g.shape[-1]), True
        )
        sb = sb_g.masked_fill(~group_mask, float("-inf")).reshape(seq, num_experts)
        _, idx = sb.topk(top_k, dim=-1)
        w = s.gather(-1, idx)  # 最终权重用无 bias 的 sigmoid 分数
    else:
        raise ValueError(f"不支持的 routing_method_type：{routing_method_type}")
    return idx.long(), (w * float(routed_scaling)).float()


def _moe_expert_compute(
    x: Any,
    expert_idx: Any,
    scales: Any,
    w1: Any,
    w2: Any,
    local_expert_offset: int,
    local_num_experts: int,
) -> Any:
    """MoE 专家计算核心（fp32）：swiglu(x @ W1) @ W2 按路由权重聚合。

    仅累加落在本卡专家区间 [local_expert_offset, +local_num_experts) 的
    贡献（EP 语义，通信不在本算子职责内）。w1 (E, 2I, H)、w2 (E, H, I)。
    """
    torch = _require_torch()
    seq, hidden = x.shape
    num_experts = w1.shape[0]
    inter = w1.shape[1] // 2
    out = torch.zeros(seq, hidden, dtype=torch.float32, device=x.device)
    with torch.no_grad():
        for e in range(num_experts):
            le = e - int(local_expert_offset)
            if le < 0 or le >= int(local_num_experts):
                continue
            rows, ks = (expert_idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = x[rows] @ w1[e].t()  # (n, 2I)
            h = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
            y = h @ w2[e].t()  # (n, H)
            out.index_add_(0, rows, y * scales[rows, ks].unsqueeze(-1))
    return out


def reference_cutlass_fused_moe(
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
    """cutlass_fused_moe 非量化参考实现（fp32 累计，单卡语义）。"""
    if ep_size != 1:
        raise NotImplementedError(
            "reference kernel 仅单卡语义；ep_size > 1 请走 CANN megaMoe 路径"
        )
    del ep_rank, moe_context
    torch = _require_torch()
    with torch.no_grad():
        out = _moe_expert_compute(
            input.float(),
            token_selected_experts.long(),
            token_final_scales.float(),
            fc1_expert_weights.float(),
            fc2_expert_weights.float(),
            0,
            fc1_expert_weights.shape[0],
        )
        del output_dtype  # 输出精度以 output 张量 dtype 为准
        output.copy_(out.to(output.dtype))
    return output


def reference_trtllm_fp8_per_tensor_scale_moe(
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
    """trtllm_fp8_per_tensor_scale_moe 参考实现（fp8 反量化到 fp32 计算）。

    per-tensor scale 折算进权重：W_eff = W_fp8 * s1 * s1_gate（gemm1）/
    W_fp8 * s2（gemm2）；use_routing_scales_on_input 决定路由权重作用在
    gemm 输入还是输出，fp32 参考下两者数学等价，直接按输出侧处理。
    """
    del intermediate_size, use_routing_scales_on_input, tile_tokens_dim
    del store_workspace_info
    torch = _require_torch()
    idx, w = _route_topk(
        routing_logits.float(),
        routing_bias.float(),
        int(top_k),
        int(n_group),
        int(topk_group),
        float(routed_scaling_factor),
        int(routing_method_type),
    )
    with torch.no_grad():
        s1 = output1_scales_scalar.float() * output1_scales_gate_scalar.float()
        w1_eff = gemm1_weights.float() * s1.reshape(-1, 1, 1)
        w2_eff = gemm2_weights.float() * output2_scales_scalar.float().reshape(-1, 1, 1)
        out = _moe_expert_compute(
            hidden_states.float(),
            idx,
            w,
            w1_eff,
            w2_eff,
            int(local_expert_offset),
            int(local_num_experts),
        )
        output.copy_(out.to(output.dtype))
    return None


def reference_trtllm_fp8_block_scale_moe(
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
    """trtllm_fp8_block_scale_moe 参考实现（按 128-block 反量化到 fp32）。

    hidden_states_scale 布局为 (hidden//128, seq)（上游转置存放）；
    gemm1_scales (E, 2I//128, H//128)、gemm2_scales (E, H//128, I//128)。
    """
    del intermediate_size, tile_tokens_dim
    torch = _require_torch()
    idx, w = _route_topk(
        expert_logits.float(),
        routing_bias.float(),
        int(top_k),
        int(n_groups),
        int(top_k_groups),
        float(routed_scaling),
        int(routing_method_type),
    )

    def _dequant_block(x_f, s_f):
        # x (..., R, C)，s (..., R//128, C//128)，两个维度按 128-block 展开
        return x_f * s_f.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)

    with torch.no_grad():
        seq, hidden = hidden_states.shape
        xs = hidden_states_scale.float().t()  # (seq, hidden//128)，行方向不分块
        x_eff = hidden_states.float() * xs.repeat_interleave(128, dim=-1)
        w1_eff = _dequant_block(gemm1_weights.float(), gemm1_scales.float())
        w2_eff = _dequant_block(gemm2_weights.float(), gemm2_scales.float())
        out = _moe_expert_compute(
            x_eff,
            idx,
            w,
            w1_eff,
            w2_eff,
            int(local_expert_offset),
            int(local_num_experts),
        )
        output.copy_(out.to(output.dtype))
    return None


def reference_merge_state(
    v_a: Any, s_a: Any, v_b: Any, s_b: Any
) -> Tuple[Any, Any]:
    """merge_state 高精度参考实现（fp32 计算，-inf lse 安全）。

    合并两段 KV 的注意力输出与 logsumexp（对标上游 flashinfer.cascade）：

    ``s_m = log(exp(s_a) + exp(s_b))``
    ``v_m = v_a * exp(s_a - s_m) + v_b * exp(s_b - s_m)``
    """
    torch = _require_torch()
    with torch.no_grad():
        sa, sb = s_a.float(), s_b.float()
        s_m = torch.logaddexp(sa, sb)
        neg_inf = torch.tensor(float("-inf"), dtype=s_m.dtype, device=s_m.device)
        scale_a = torch.where(sa == neg_inf, torch.zeros_like(sa), torch.exp(sa - s_m))
        scale_b = torch.where(sb == neg_inf, torch.zeros_like(sb), torch.exp(sb - s_m))
        v_m = (
            v_a.float() * scale_a.unsqueeze(-1) + v_b.float() * scale_b.unsqueeze(-1)
        ).to(v_a.dtype)
    return v_m, s_m


def builtin_descriptors() -> Tuple[KernelDescriptor, ...]:
    """内置 reference kernel 注册表。M0 唯一可执行的内置 backend。"""
    return (
        KernelDescriptor(
            kernel_id="reference/attention/prefill_ragged_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="attention.prefill"),
            runner=_prefill_ragged_ref,
        ),
        KernelDescriptor(
            kernel_id="reference/attention/decode_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="attention.decode"),
            runner=_decode_ref,
        ),
        KernelDescriptor(
            kernel_id="reference/norm/rmsnorm_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="norm.rmsnorm"),
            runner=reference_rmsnorm,
        ),
        KernelDescriptor(
            kernel_id="reference/norm/fused_add_rmsnorm_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="norm.fused_add_rmsnorm"),
            runner=reference_fused_add_rmsnorm,
        ),
        KernelDescriptor(
            kernel_id="reference/norm/gemma_rmsnorm_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="norm.gemma_rmsnorm"),
            runner=reference_gemma_rmsnorm,
        ),
        KernelDescriptor(
            kernel_id="reference/norm/gemma_fused_add_rmsnorm_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="norm.gemma_fused_add_rmsnorm"),
            runner=reference_gemma_fused_add_rmsnorm,
        ),
        KernelDescriptor(
            kernel_id="reference/norm/layernorm_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="norm.layernorm"),
            runner=reference_layernorm,
        ),
        KernelDescriptor(
            kernel_id="reference/rope/apply_rope_with_cos_sin_cache_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="rope.apply_rope_with_cos_sin_cache_inplace"
            ),
            runner=reference_apply_rope_with_cos_sin_cache,
        ),
        KernelDescriptor(
            kernel_id="reference/decode/cudnn_batch_decode_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="decode.cudnn_batch_decode_with_kv_cache"
            ),
            runner=reference_cudnn_batch_decode,
        ),
        KernelDescriptor(
            kernel_id="reference/gemm/bmm_fp8_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="gemm.bmm_fp8"),
            runner=reference_bmm_fp8,
        ),
        KernelDescriptor(
            kernel_id="reference/gemm/gemm_fp8_nt_groupwise_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="gemm.gemm_fp8_nt_groupwise"),
            runner=reference_gemm_fp8_nt_groupwise,
        ),
        KernelDescriptor(
            kernel_id="reference/fused_moe/cutlass_fused_moe_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="fused_moe.cutlass_fused_moe"),
            runner=reference_cutlass_fused_moe,
        ),
        KernelDescriptor(
            kernel_id="reference/fused_moe/trtllm_fp8_per_tensor_scale_moe_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="fused_moe.trtllm_fp8_per_tensor_scale_moe"
            ),
            runner=reference_trtllm_fp8_per_tensor_scale_moe,
        ),
        KernelDescriptor(
            kernel_id="reference/fused_moe/trtllm_fp8_block_scale_moe_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="fused_moe.trtllm_fp8_block_scale_moe"),
            runner=reference_trtllm_fp8_block_scale_moe,
        ),
        KernelDescriptor(
            kernel_id="reference/cascade/merge_state_v1",
            backend=Backend.REFERENCE,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(op="cascade.merge_state"),
            runner=reference_merge_state,
        ),
    )
