"""decode 对外 API 门面（对标 flashinfer 顶层模块，§2.3、§3.1）。

一次性接口（``single_decode_with_kv_cache``）与 plan/run 两段式 wrapper
（``BatchDecodeWithPagedKVCacheWrapper``）共用同一条
"构建签名 → dispatch → 拉起"主链路。
"""

from __future__ import annotations

from typing import Any, Optional, Union

from .runtime import (
    Backend,
    InputSignature,
    Plan,
    build_signature,
    current_hardware,
    dispatch,
    get_global_registry,
)
from .prefill import _run_selected

_OP_DECODE = "attention.decode"


def single_decode_with_kv_cache_with_jit_module(
    jit_module, q, k, v, *args, kv_layout="NHD", window_left=-1, return_lse=False,
):
    raise NotImplementedError("T2.2: single decode JIT facade is not implemented")


def single_decode_with_kv_cache(
    q,
    k_cache,
    v_cache,
    kv_layout: str = "NHD",
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """单步 decode attention：q (B, H, D)，k/v_cache (B, S, H_kv, D)。

    调度参数语义同 prefill（§3.2.2）。
    """
    signature = build_signature(
        _OP_DECODE,
        (q, k_cache, v_cache),
        layouts=(kv_layout, kv_layout, kv_layout),
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    return _run_selected(decision.kernel_id, q, k_cache, v_cache)


class BatchDecodeWithPagedKVCacheWrapper:
    """paged KV decode 的 plan/run wrapper（对标 flashinfer，§3.3）。

    - ``plan()``：按 paged 结构参数构建签名、dispatch、固化 Plan（含选定
      kernel），只发生一次；
    - ``run()``：复用 Plan 直接拉起 kernel，不重复 dispatch。
    """

    def __init__(
        self,
        float_workspace_buffer: Any,
        kv_layout: str = "NHD",
        *,
        backend: Union[str, Backend, None] = None,
        kernel_id: Optional[str] = None,
    ) -> None:
        self._kv_layout = kv_layout
        self._backend = backend
        self._kernel_id = kernel_id
        self._plan: Optional[Plan] = None

    @property
    def is_planning_done(self) -> bool:
        return self._plan is not None

    def plan(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_dtype: Any = None,
        kv_dtype: Any = None,
        causal: bool = True,
    ) -> Plan:
        """paged 结构参数期 dispatch：真实 shape 未定，以结构维度构建签名。"""
        def _dtype_name(dtype: Any) -> str:
            name = str(dtype) if dtype is not None else "unset"
            return name[len("torch."):] if name.startswith("torch.") else name

        hardware = current_hardware()
        signature = InputSignature(
            op=_OP_DECODE,
            dtypes=(_dtype_name(q_dtype), _dtype_name(kv_dtype or q_dtype)),
            shapes=(),
            layouts=(self._kv_layout,),
            specializations=tuple(sorted((
                ("causal", str(bool(causal))),
                ("head_dim", str(int(head_dim))),
                ("num_kv_heads", str(int(num_kv_heads))),
                ("num_qo_heads", str(int(num_qo_heads))),
                ("page_size", str(int(page_size))),
                ("paged", "True"),
            ))),
        )
        decision = dispatch(
            signature,
            hardware,
            backend=self._backend,
            kernel_id=self._kernel_id,
        )
        self._plan = Plan(
            signature=signature, hardware=hardware, decision=decision
        )
        return self._plan

    def run(self, q, kv_cache, *args, **kwargs):
        """复用 plan 期固化的 kernel 执行。plan() 未调用时显式报错。"""
        if self._plan is None:
            raise RuntimeError("plan() 未调用：wrapper 无已固化的 Plan")
        return _run_selected(self._plan.kernel_id, q, kv_cache, *args, **kwargs)


_OP_CUDNN_DECODE = "decode.cudnn_batch_decode_with_kv_cache"


def cudnn_batch_decode_with_kv_cache(
    q,
    k_cache,
    v_cache,
    scale: float,
    workspace_buffer,
    *,
    max_sequence_kv: int,
    actual_seq_lens_kv=None,
    block_tables=None,
    is_cuda_graph_compatible: bool = False,
    batch_offsets_q=None,
    batch_offsets_o=None,
    batch_offsets_k=None,
    batch_offsets_v=None,
    out=None,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """批量 paged KV decode attention（对标 flashinfer.decode 同名接口）。

    参数名、默认值与语义对齐上游；仅 causal；支持 GQA（H_qo 为 H_kv
    整数倍）。返回 (batch_size, num_heads_qo, head_dim)。

    与上游的显式差异（parity 清单标记 compatible）：
    - ``workspace_buffer`` / ``is_cuda_graph_compatible`` 为 cuDNN/CUDA Graph
      专属参数，NPU 无对应机制，接受并忽略；
    - ``actual_seq_lens_kv`` / ``block_tables`` 上游默认 ``None`` 但 kernel
      实际必需，本地显式校验非空；
    - ``batch_offsets_*``（上游 CUDA Graph 变长批偏移）暂不支持，非空时
      显式报错而非静默忽略。
    """
    del workspace_buffer, is_cuda_graph_compatible  # NPU 无对应机制，接受并忽略
    if any(
        x is not None
        for x in (batch_offsets_q, batch_offsets_o, batch_offsets_k, batch_offsets_v)
    ):
        raise NotImplementedError("batch_offsets_* 暂不支持（上游 CUDA Graph 专属）")
    if actual_seq_lens_kv is None or block_tables is None:
        raise ValueError("actual_seq_lens_kv 与 block_tables 不能为空")

    import torch  # 延迟导入：模块本身不依赖 torch（§14.1 host 单测）

    if out is None:
        out = torch.empty(
            q.shape[0], q.shape[1], v_cache.shape[-1],
            device=q.device, dtype=q.dtype,
        )
    signature = build_signature(
        _OP_CUDNN_DECODE,
        (q, k_cache, v_cache, actual_seq_lens_kv, block_tables),
        specializations={
            "scale": float(scale),
            "max_sequence_kv": int(max_sequence_kv),
        },
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(
        decision.kernel_id,
        q,
        k_cache,
        v_cache,
        float(scale),
        int(max_sequence_kv),
        actual_seq_lens_kv,
        block_tables,
        out,
    )
    return out
