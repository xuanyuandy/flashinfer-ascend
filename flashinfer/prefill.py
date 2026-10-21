"""prefill 对外 API 门面（对标 flashinfer 顶层模块，§2.3、§3.1）。

门面只做三件事：构建输入签名 → 委托 dispatcher 选择 kernel → 拉起执行。
一行 kernel 代码都没有（§2.3：API 文件 = 壳，kernels/ = 芯）。
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

_OP_RAGGED_PREFILL = "attention.prefill"


class BatchPrefillWithPagedKVCacheWrapper:
    """Public declaration; the executable facade is tracked by task T2.2."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("T2.2: paged prefill facade is not implemented")


class BatchPrefillWithRaggedKVCacheWrapper:
    """Public declaration; the executable facade is tracked by task T2.2."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("T2.2: ragged prefill facade is not implemented")


def single_prefill_with_kv_cache_with_jit_module(
    jit_module, q, k, v, *args, kv_layout="NHD", mask_mode=0,
    window_left=-1, return_lse=False,
):
    raise NotImplementedError("T2.2: single prefill JIT facade is not implemented")


def _run_selected(kernel_id: str, *args, **kwargs):
    """拉起 dispatch 选中的 kernel。产物未接入时显式报错（M0 骨架）。"""
    descriptor = get_global_registry().get(kernel_id)
    if descriptor.runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入"
            "（AOT 产物加载 / JIT 编译拉起随 M1 起接入，§15）" % kernel_id
        )
    return descriptor.runner(*args, **kwargs)


def single_prefill_with_kv_cache(
    q,
    k,
    v,
    causal: bool = False,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """单请求 ragged prefill attention。

    调度参数（§3.2.2，与计算语义正交）：
    - ``backend``   显式限定 backend 组（默认 None → 全局配置，缺省 auto）；
    - ``kernel_id`` 显式锁定唯一 kernel（优先级最高，自定义 kernel 走此入口）。
    """
    signature = build_signature(
        _OP_RAGGED_PREFILL,
        (q, k, v),
        specializations={"causal": bool(causal)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    return _run_selected(decision.kernel_id, q, k, v, causal=causal)
