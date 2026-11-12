"""reference backend：高精度参考实现（§5.2）。

定位：correctness 基线与调试对照，**默认不进生产热路径**——
参与 dispatch 需显式 opt-in（§12）。M0 阶段只提供朴素 attention
参考实现，正确性细节随 M1 起逐步完善。
"""

from __future__ import annotations

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

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

def _reference_rope_core(
    q: Any, k: Any, pos_ids: Any, rotary_dim, interleave: bool,
    rope_scale: float, rope_theta: float, inplace: bool,
):
    """``(nnz, heads, head_dim)`` 布局 RoPE 的 float64 参考实现。

    频率按上游 ``pos_enc.cuh`` 的 plain-rope 分支（smooth=0）：
    ``freq[j] = rope_theta ** (-2j / rotary_dim) / rope_scale``。
    ``interleave=True`` 旋转偶/奇维，``False`` 旋转前后半——注意与
    ``is_neox`` 取值相反。
    """
    torch = _require_torch()
    rd = int(rotary_dim) if rotary_dim is not None else int(q.shape[-1])
    half = rd // 2
    with torch.no_grad():
        j = torch.arange(0, rd, 2, device=pos_ids.device, dtype=torch.float64)
        inv_freq = torch.pow(
            torch.tensor(float(rope_theta), dtype=torch.float64, device=pos_ids.device),
            -j / rd,
        ) / float(rope_scale)
        angle = pos_ids.to(torch.float64).unsqueeze(-1) * inv_freq  # (nnz, rd/2)
        cos = angle.cos().unsqueeze(1)  # (nnz, 1, rd/2)
        sin = angle.sin().unsqueeze(1)
        outs = []
        for t in (q, k):
            dst = t if inplace else t.clone()
            x = dst[..., :rd].double()
            if interleave:
                xe, xo = x[..., 0::2], x[..., 1::2]
                rot = torch.stack(
                    [xe * cos - xo * sin, xo * cos + xe * sin], dim=-1
                ).flatten(-2)
            else:
                x1, x2 = x[..., :half], x[..., half:]
                rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
            dst[..., :rd].copy_(rot.to(t.dtype))
            outs.append(dst)
    return None if inplace else (outs[0], outs[1])

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
    from flashinfer.tllm_enums import RoutingMethodType

    try:
        method = RoutingMethodType(routing_method_type)
    except ValueError as error:
        raise ValueError(f"unknown routing_method_type: {routing_method_type}") from error
    if method.value > 4:
        raise NotImplementedError(f"{method!r} routing is not implemented by this provider")
    torch = _require_torch()
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
