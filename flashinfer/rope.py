"""rope 对外 API 门面（对标 flashinfer.rope，§2.3、§3.1）。

函数名、参数名、默认值与语义对齐上游（flashinfer-ai/flashinfer
flashinfer/rope.py）。门面只做三件事：构建输入签名 → 委托 dispatcher
选择 kernel → 拉起执行。一行 kernel 代码都没有（§2.3）。
"""

from __future__ import annotations

from typing import Optional, Union

from .runtime import Backend, build_signature, current_hardware, dispatch

_OP_APPLY_ROPE = "rope.apply_rope_with_cos_sin_cache_inplace"


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
