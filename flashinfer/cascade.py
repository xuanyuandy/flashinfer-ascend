"""cascade 对外 API 门面（对标 flashinfer.cascade，§2.3、§3.1）。

函数名、参数名、默认值与语义对齐上游（flashinfer-ai/flashinfer
flashinfer/cascade.py）。门面只做三件事：构建输入签名 → 委托 dispatcher
选择 kernel → 拉起执行。一行 kernel 代码都没有（§2.3）。
"""

from __future__ import annotations

from typing import Optional, Union

from .runtime import (
    Backend,
    build_signature,
    current_hardware,
    dispatch,
)
from .runtime.runner import run_selected as _run_selected

_OP_MERGE_STATE = "cascade.merge_state"


def merge_states(v, s):
    """Merge stacked states with base-2 LSE; v [seq, states, heads, dim]."""
    signature = build_signature("cascade.merge_states", (v, s))
    decision = dispatch(signature, current_hardware(), backend=None)
    return _run_selected(decision.kernel_id, v, s)


def merge_state(
    v_a,
    s_a,
    v_b,
    s_b,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    r"""Merge the attention output ``V`` and the logsumexp value ``S`` from the
    two KV-segments.

    Parameters
    ----------
    v_a : torch.Tensor
        The attention output from the KV segment ``A``, shape:
        ``[seq_len, num_heads, head_dim]``。
    s_a : torch.Tensor
        The logsumexp value from the KV segment ``A``，float32，shape:
        ``[seq_len, num_heads]``。
    v_b : torch.Tensor
        The attention output from the KV segment ``B``, shape 同 ``v_a``。
    s_b : torch.Tensor
        The logsumexp value from the KV segment ``B``, shape 同 ``s_a``。
    backend / kernel_id
        调度覆盖参数（§3.2.2，与计算语义正交）。

    Returns
    -------
    V : torch.Tensor
        The merged attention output，shape 同 ``v_a``。
    S : torch.Tensor
        The logsumexp value from the merged KV-segment，float32，shape 同
        ``s_a``。

    本仓现有 ``merge_state`` 暂沿用自然对数 LSE，已在账本登记为有意差异。
    新的 ``merge_states`` 使用上游 log2 契约；两者不可直接混用 LSE。
    """
    if v_a.shape != v_b.shape:
        raise ValueError(
            f"merge_state: v_a 与 v_b 形状须一致，got {tuple(v_a.shape)} vs "
            f"{tuple(v_b.shape)}"
        )
    if v_a.ndim != 3:
        raise ValueError(f"merge_state: v 须为 3D (seq_len, num_heads, head_dim)")
    if s_a.shape != v_a.shape[:2] or s_b.shape != v_a.shape[:2]:
        raise ValueError(
            "merge_state: s 形状须为 (seq_len, num_heads)，与 v 前两维一致"
        )
    signature = build_signature(
        _OP_MERGE_STATE,
        (v_a, s_a, v_b, s_b),
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    return _run_selected(decision.kernel_id, v_a, s_a, v_b, s_b)
