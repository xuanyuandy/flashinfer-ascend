"""Execution adapter for cascade.merge_state (reference)."""

from __future__ import annotations
from flashinfer.reference._shared import (Any, Tuple, _require_torch)

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

run = reference_merge_state
