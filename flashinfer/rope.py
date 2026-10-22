"""rope 对外 API 门面（对标 flashinfer.rope，§2.3、§3.1）。

函数名、参数名、默认值与语义对齐上游（flashinfer-ai/flashinfer
flashinfer/rope.py）。门面只做三件事：构建输入签名 → 委托 dispatcher
选择 kernel → 拉起执行。一行 kernel 代码都没有（§2.3）。
"""

from __future__ import annotations

from typing import Optional, Union

from .runtime import Backend, build_signature, current_hardware, dispatch

_OP_APPLY_ROPE = "rope.apply_rope_with_cos_sin_cache_inplace"
_OP_APPLY_ROPE_OUT = "rope.apply_rope_with_cos_sin_cache"
_OP_POS_IDS_INPLACE = "rope.apply_rope_pos_ids_inplace"
_OP_POS_IDS = "rope.apply_rope_pos_ids"
_OP_ROPE_INPLACE = "rope.apply_rope_inplace"
_OP_ROPE = "rope.apply_rope"


def apply_rope_with_cos_sin_cache_inplace(
    positions,
    query,
    key,
    head_size: int,
    cos_sin_cache,
    is_neox: bool = True,
    *,
    backend: Optional[Union[str, Backend]] = None,
    kernel_id: Optional[str] = None,
) -> None:
    """对 query/key 原地施加旋转位置编码（RoPE），使用预计算的 cos/sin 缓存。

    与上游 ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace``
    参数名、默认值、原地语义一致（兼容 SGL/vLLM 用法），无返回值。

    ``cos_sin_cache`` shape 为 ``(max_seq_len, rotary_dim)``，前一半为
    cos、后一半为 sin，dtype 必须为 float32（与上游校验一致）；
    每个 head 仅旋转前 ``rotary_dim`` 维，其余维保持不变。

    Parameters
    ----------
    positions
        位置索引，shape ``(nnz,)``。
    query
        query 张量，shape ``(nnz, num_q_heads * head_size)``，原地修改。
    key
        key 张量，shape ``(nnz, num_k_heads * head_size)``，原地修改。
    head_size
        单头维度。
    cos_sin_cache
        cos/sin 缓存，shape ``(max_seq_len, rotary_dim)``，float32。
    is_neox
        ``True`` 为 GPT-NeoX 半旋转（默认）；``False`` 为 GPT-J 交错旋转。

    Notes
    -----
    本框架扩展的关键字参数 ``backend`` / ``kernel_id`` 位于上游签名之后，
    drop-in 替换不受影响；``backend="auto"`` 为默认策略（CANN 优先）。
    """
    dtype_name = str(cos_sin_cache.dtype).replace("torch.", "")
    if dtype_name != "float32":
        raise ValueError("cos_sin_cache should be float32")

    signature = build_signature(
        _OP_APPLY_ROPE,
        (positions, query, key, cos_sin_cache),
        specializations={"head_size": int(head_size), "is_neox": bool(is_neox)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )

    from .norm import _run_selected

    _run_selected(
        decision.kernel_id,
        positions,
        query,
        key,
        int(head_size),
        cos_sin_cache,
        bool(is_neox),
    )


def _pos_ids_from_indptr(indptr, offsets):
    """由 ``indptr``（batch+1）与 ``offsets``（batch）展开逐 token 位置。

    第 i 段是 ``[indptr[i], indptr[i+1])``，其位置为
    ``offsets[i] + 0..seg_len-1``（与上游 ragged 语义一致）。
    纯 host 侧索引运算，不是 kernel。
    """
    import torch

    idx = indptr.to(torch.int64)
    off = offsets.to(torch.int64)
    lens = idx[1:] - idx[:-1]
    total = int(lens.sum().item())
    seg = torch.repeat_interleave(torch.arange(lens.numel(), device=idx.device), lens)
    starts = torch.repeat_interleave(idx[:-1], lens)
    within = torch.arange(total, device=idx.device) - starts
    return (off[seg] + within).to(torch.int32)


def _dispatch_rope(op, tensors, specializations, args, backend, kernel_id):
    signature = build_signature(op, tensors, specializations=specializations)
    decision = dispatch(signature, current_hardware(), backend=backend, kernel_id=kernel_id)
    from .norm import _run_selected

    return _run_selected(decision.kernel_id, *args)


def apply_rope_pos_ids_inplace(
    q, k, pos_ids, rotary_dim: Optional[int] = None, interleave: bool = False,
    rope_scale: float = 1, rope_theta: float = 1e4,
    *, backend: Optional[Union[str, Backend]] = None, kernel_id: Optional[str] = None,
) -> None:
    """对 ragged 布局的 q/k 原地施加 RoPE，位置由 ``pos_ids`` 显式给出。

    与上游 ``flashinfer.rope.apply_rope_pos_ids_inplace`` 参数名、默认值、
    原地语义一致。``q``/``k`` shape 为 ``(nnz, num_heads, head_dim)``。

    ``interleave=True`` 旋转偶/奇维（GPT-J 风格）；``False`` 旋转前后半
    （GPT-NeoX 风格）。注意与 ``apply_rope_with_cos_sin_cache_inplace``
    的 ``is_neox`` 取值相反。
    """
    _dispatch_rope(
        _OP_POS_IDS_INPLACE, (q, k, pos_ids),
        {"interleave": bool(interleave), "rotary_dim": int(rotary_dim) if rotary_dim else 0},
        (q, k, pos_ids, rotary_dim, bool(interleave), float(rope_scale), float(rope_theta)),
        backend, kernel_id,
    )


def apply_rope_pos_ids(
    q, k, pos_ids, rotary_dim: Optional[int] = None, interleave: bool = False,
    rope_scale: float = 1, rope_theta: float = 1e4,
    *, backend: Optional[Union[str, Backend]] = None, kernel_id: Optional[str] = None,
):
    """``apply_rope_pos_ids_inplace`` 的非原地版本，返回 ``(q_rope, k_rope)``。"""
    return _dispatch_rope(
        _OP_POS_IDS, (q, k, pos_ids),
        {"interleave": bool(interleave), "rotary_dim": int(rotary_dim) if rotary_dim else 0},
        (q, k, pos_ids, rotary_dim, bool(interleave), float(rope_scale), float(rope_theta)),
        backend, kernel_id,
    )


def apply_rope_inplace(
    q, k, indptr, offsets, rotary_dim: Optional[int] = None, interleave: bool = False,
    rope_scale: float = 1, rope_theta: float = 1e4,
    *, backend: Optional[Union[str, Backend]] = None, kernel_id: Optional[str] = None,
) -> None:
    """按 ``indptr``/``offsets`` 描述的分段 ragged 批次原地施加 RoPE。

    与上游 ``flashinfer.rope.apply_rope_inplace`` 一致：第 i 段为
    ``q[indptr[i]:indptr[i+1]]``，其起始位置为 ``offsets[i]``。
    """
    pos_ids = _pos_ids_from_indptr(indptr, offsets)
    _dispatch_rope(
        _OP_ROPE_INPLACE, (q, k, pos_ids),
        {"interleave": bool(interleave), "rotary_dim": int(rotary_dim) if rotary_dim else 0},
        (q, k, pos_ids, rotary_dim, bool(interleave), float(rope_scale), float(rope_theta)),
        backend, kernel_id,
    )


def apply_rope(
    q, k, indptr, offsets, rotary_dim: Optional[int] = None, interleave: bool = False,
    rope_scale: float = 1, rope_theta: float = 1e4,
    *, backend: Optional[Union[str, Backend]] = None, kernel_id: Optional[str] = None,
):
    """``apply_rope_inplace`` 的非原地版本，返回 ``(q_rope, k_rope)``。"""
    pos_ids = _pos_ids_from_indptr(indptr, offsets)
    return _dispatch_rope(
        _OP_ROPE, (q, k, pos_ids),
        {"interleave": bool(interleave), "rotary_dim": int(rotary_dim) if rotary_dim else 0},
        (q, k, pos_ids, rotary_dim, bool(interleave), float(rope_scale), float(rope_theta)),
        backend, kernel_id,
    )


def apply_rope_with_cos_sin_cache(
    positions, query, key, head_size: int, cos_sin_cache, is_neox: bool = True,
    *, backend: Optional[Union[str, Backend]] = None, kernel_id: Optional[str] = None,
):
    """``apply_rope_with_cos_sin_cache_inplace`` 的非原地版本。

    返回 ``(query_rope, key_rope)``，输入不被修改；其余语义与原地版一致。
    """
    dtype_name = str(cos_sin_cache.dtype).replace("torch.", "")
    if dtype_name != "float32":
        raise ValueError("cos_sin_cache should be float32")
    return _dispatch_rope(
        _OP_APPLY_ROPE_OUT, (positions, query, key, cos_sin_cache),
        {"head_size": int(head_size), "is_neox": bool(is_neox)},
        (positions, query, key, int(head_size), cos_sin_cache, bool(is_neox)),
        backend, kernel_id,
    )
