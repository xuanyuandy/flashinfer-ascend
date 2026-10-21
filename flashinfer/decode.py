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
    from ._single_attention import run_single_jit
    from .attention.schema import AttentionMode

    return run_single_jit(AttentionMode.SINGLE_DECODE, jit_module, q, k, v, args,
                          kv_layout=kv_layout, window_left=window_left,
                          return_lse=return_lse)


def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    """Decode one request: q [heads, dim], K/V [tokens, kv_heads, dim]."""
    from ._single_attention import single_attention
    from .attention.schema import AttentionMode

    return single_attention(
        AttentionMode.SINGLE_DECODE, q, k, v, kv_layout=kv_layout,
        pos_encoding_mode=pos_encoding_mode, use_tensor_cores=use_tensor_cores,
        scale_q=q_scale, k_scale=k_scale, v_scale=v_scale,
        window_left=window_left, logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale, rope_scale=rope_scale, rope_theta=rope_theta,
        return_lse=return_lse,
    )


from .attention.public_batch import PublicBatchWrapper
from .attention.schema import AttentionMode
from functools import partialmethod


class BatchDecodeWithPagedKVCacheWrapper(PublicBatchWrapper):
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        self._init_facade(AttentionMode.BATCH_DECODE_PAGED, locals())

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *deprecated_positional_args: Any,
        **kwargs: Any,
    ) -> None:
        self._plan_decode(locals())

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        kv_cache_sf: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._run_facade(q, paged_kv_cache, locals())

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        return self._deprecated_forward(q, paged_kv_cache, locals())

    def workspace_size(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
        q_len_per_req: int = 1,
    ) -> Tuple[int, int]:
        return self._workspace_query(locals())

    run_return_lse = partialmethod(run, return_lse=True)
    begin_forward = plan

    @property
    def use_tensor_cores(self):
        return self._use_tensor_cores

    @property
    def use_matrix_cores(self):
        return self._use_tensor_cores

    def end_forward(self):
        return None


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
