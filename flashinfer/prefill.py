"""prefill 对外 API 门面（对标 flashinfer 顶层模块，§2.3、§3.1）。

门面只做三件事：构建输入签名 → 委托 dispatcher 选择 kernel → 拉起执行。
设备源码位于 csrc/kernels/ascendc/；本文件不包含 kernel 实现。
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


def single_prefill_with_kv_cache_with_jit_module(
    jit_module, q, k, v, *args, kv_layout="NHD", mask_mode=0,
    window_left=-1, return_lse=False,
):
    from ._single_attention import run_single_jit
    from .attention.schema import AttentionMode

    return run_single_jit(AttentionMode.SINGLE_PREFILL, jit_module, q, k, v, args,
                          kv_layout=kv_layout, mask_mode=mask_mode,
                          window_left=window_left, return_lse=return_lse)


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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: bool = False,
    kv_cache_sf: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    from ._single_attention import single_attention
    from .attention.schema import AttentionMode

    return single_attention(
        AttentionMode.SINGLE_PREFILL, q, k, v, backend=backend,
        kv_layout=kv_layout, pos_encoding_mode=pos_encoding_mode,
        causal=causal, scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
        k_scale=k_scale, v_scale=v_scale, o_dtype=o_dtype,
        custom_mask=custom_mask, packed_custom_mask=packed_custom_mask,
        sm_scale=sm_scale, window_left=window_left,
        logits_soft_cap=logits_soft_cap, rope_scale=rope_scale,
        rope_theta=rope_theta, use_fp16_qk_reduction=use_fp16_qk_reduction,
        return_lse=return_lse, kv_cache_sf=kv_cache_sf,
    )


from .attention.public_batch import PublicBatchWrapper
from .attention.schema import AttentionMode
from functools import partialmethod


class BatchPrefillWithPagedKVCacheWrapper(PublicBatchWrapper):
    """Paged prefill; graph plans copy into caller-owned fixed metadata buffers.

    plan() runs outside capture. The first successful graph plan freezes the
    execution choices; aclgraph capture/replay belongs to the caller.
    use_max_workspace=True queries FIA's maximum in graph plan() and checks
    the caller's fixed byte workspace. The default False preserves memory use.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
        jit_kwargs: Optional[Dict[str, Any]] = None,
        variant_owns_mask: bool = False,
        max_total_num_rows: Optional[int] = None,
        *,
        use_max_workspace: bool = False,
    ) -> None:
        self._init_facade(AttentionMode.BATCH_PREFILL_PAGED, locals())

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
        uniform_q_len: Optional[int] = None,
    ) -> None:
        self._plan_prefill(locals())

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[Union[float, torch.Tensor]] = None,
        k_scale: Optional[Union[float, torch.Tensor]] = None,
        v_scale: Optional[Union[float, torch.Tensor]] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        lse_base: str = "log2",
        lse_layout: str = "NH",
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        kv_cache_sf: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        use_fp16_softmax: Optional[bool] = None,
        uses_spcompress: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._run_facade(q, paged_kv_cache, locals())

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
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
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
        uniform_q_len: Optional[int] = None,
    ) -> Tuple[int, int]:
        return self._workspace_query(locals())

    run_return_lse = partialmethod(run, return_lse=True)
    begin_forward = plan

    def end_forward(self):
        return None


class BatchPrefillWithRaggedKVCacheWrapper(PublicBatchWrapper):
    """Ragged prefill with paged prefill's graph and use_max_workspace contracts."""

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
        jit_kwargs: Optional[Dict[str, Any]] = None,
        variant_owns_mask: bool = False,
        *,
        use_max_workspace: bool = False,
    ) -> None:
        self._init_facade(AttentionMode.BATCH_PREFILL_RAGGED, locals())

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_q: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        v_indptr: Optional[torch.Tensor] = None,
        o_indptr: Optional[torch.Tensor] = None,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_indptr_cpu: Optional[torch.Tensor] = None,
        uniform_q_len: Optional[int] = None,
    ) -> None:
        self._plan_prefill(locals())

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        lse_base: str = "log2",
        lse_layout: str = "NH",
        enable_pdl: Optional[bool] = None,
        kv_cache_sf: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._run_facade(q, (k, v), locals())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        return self._deprecated_forward(q, (k, v), locals())

    run_return_lse = partialmethod(run, return_lse=True)
    begin_forward = plan

    def end_forward(self):
        return None
