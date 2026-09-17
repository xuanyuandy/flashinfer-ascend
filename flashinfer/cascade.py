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
    get_global_registry,
)
from .runtime.schema import DispatchError

_OP_MERGE_STATE = "cascade.merge_state"


def _run_selected(kernel_id: str, *args, **kwargs):
    """拉起 dispatch 选中的 kernel。产物未接入时显式报错。"""
    descriptor = get_global_registry().get(kernel_id)
    if descriptor.runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入"
            "（AOT 产物加载 / JIT 编译拉起随后续里程碑接入，§15）" % kernel_id
        )
    return descriptor.runner(*args, **kwargs)


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
