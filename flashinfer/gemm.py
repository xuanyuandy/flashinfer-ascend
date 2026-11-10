"""GEMM 系列对外 API（§3.1 薄门面层）。

对齐上游 flashinfer.gemm 的 fp8 GEMM 接口：
- ``bmm_fp8``：fp8 per-tensor scale 批量矩阵乘；
- ``gemm_fp8_nt_groupwise``：fp8 分组 scale NT 布局矩阵乘。

硬件支持说明（§12 验收口径）：
fp8 矩阵乘在 CANN 中仅 950 代际（A5）具备原生算子；910B 上
``npu_quant_matmul`` 仅支持 int8/int4，且 torch float8 张量无法物化到
NPU。因此 CANN kernel 限定 950 注册（``npu_grouped_matmul`` fp8 路径，
见 cann/gemm.py），910B 环境 auto 无候选、可显式 opt-in reference
kernel（CPU 反量化到 fp32 计算），保证注册、调度与门面链路可用。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from .runtime import (
    Backend,
    build_signature,
    current_hardware,
    dispatch,
)
from .runtime.runner import run_selected as _run_selected

_OP_BMM_FP8 = "gemm.bmm_fp8"
_OP_GEMM_FP8_NT_GROUPWISE = "gemm.gemm_fp8_nt_groupwise"


def _dtype_name(dtype) -> str:
    name = str(dtype)
    return name[len("torch."):] if name.startswith("torch.") else name


def bmm_fp8(
    A,
    B,
    A_scale,
    B_scale,
    dtype,
    out=None,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """BMM FP8（对标 flashinfer.gemm.bmm_fp8）。

    ``out = (A @ B) * A_scale * B_scale``。

    Parameters
    ----------
    A : torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3 或 fp8 e5m2。
    B : torch.Tensor
        Mat2 tensor, shape (b, k, n)（上游要求 column major），fp8。
    A_scale / B_scale : torch.Tensor
        per-tensor scale，float。
    dtype : torch.dtype
        输出 dtype，bf16 或 fp16。
    out : Optional[torch.Tensor]
        输出张量 (b, m, n)，缺省时分配。
    backend / kernel_id
        调度覆盖参数（§3.2.2，与计算语义正交）。

    Returns
    -------
    output : torch.Tensor
        shape (b, m, n)，dtype 由 ``dtype`` 指定。
    """
    import torch  # 延迟导入：模块本身不依赖 torch（§14.1 host 单测）

    if out is None:
        out = torch.empty(
            A.shape[0], A.shape[1], B.shape[2], device=A.device, dtype=dtype
        )
    signature = build_signature(
        _OP_BMM_FP8,
        (A, B, A_scale, B_scale),
        specializations={"dtype": _dtype_name(dtype)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(decision.kernel_id, A, B, A_scale, B_scale, out)
    return out


def gemm_fp8_nt_groupwise(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode: str = "MN",
    mma_sm: int = 1,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out=None,
    out_dtype=None,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """fp8 分组 scale NT 布局矩阵乘（对标 flashinfer.gemm 同名接口）。

    ``out = (a ⊙ sa) @ (b ⊙ sb)^T``，sa/sb 按 (gm, gn, gk) 粒度分组。

    Parameters
    ----------
    a : torch.Tensor
        行主序输入 (m, k)，fp8 e4m3 或 fp8 e5m2。
    b : torch.Tensor
        列主序输入 (n, k)（NT 布局），fp8。
    a_scale / b_scale : torch.Tensor
        分组 scale：scale_major_mode="K" 时形状 (m//gm, k//gk) /
        (n//gn, k//gk)；"MN" 时转置存放。
    scale_major_mode : str
        scale 张量布局，"MN" 或 "K"。
    mma_sm : int
        NVIDIA SM100 双 SM MMA 专属参数，NPU 无对应机制，接受并忽略。
    scale_granularity_mnk : Tuple[int, int, int]
        分组粒度 (gm, gn, gk)。
    out / out_dtype
        输出张量 (m, n) 及其 dtype（缺省 bf16）。
    backend / kernel_id
        调度覆盖参数（§3.2.2，与计算语义正交）。

    Returns
    -------
    output : torch.Tensor
        shape (m, n)。
    """
    del mma_sm  # NPU 无对应机制，接受并忽略（签名级 drop-in 兼容）
    import torch  # 延迟导入：模块本身不依赖 torch（§14.1 host 单测）

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Shape mismatch. a.shape = {a.shape}, b.shape = {b.shape}"
        )
    if scale_major_mode not in ("MN", "K"):
        raise ValueError(f"scale_major_mode 仅支持 'MN'/'K'：{scale_major_mode}")
    if out is None:
        out = torch.empty(
            a.shape[0], b.shape[0],
            device=a.device, dtype=out_dtype or torch.bfloat16,
        )
    signature = build_signature(
        _OP_GEMM_FP8_NT_GROUPWISE,
        (a, b, a_scale, b_scale),
        specializations={
            "scale_major_mode": str(scale_major_mode),
            "scale_granularity_mnk": str(tuple(scale_granularity_mnk)),
        },
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(
        decision.kernel_id,
        a,
        b,
        a_scale,
        b_scale,
        str(scale_major_mode),
        tuple(scale_granularity_mnk),
        out,
    )
    return out


_OP_MM_BF16 = "gemm.mm_bf16"
_OP_BMM_BF16 = "gemm.bmm_bf16"
_OP_SEGMENT_GEMM = "gemm.segment_gemm"


def matmul(
    a, b, out=None, *, backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """Compute ``out = a @ b`` through CANN, Ascend C AOT, or Triton JIT.

    All backends accept identical contiguous ND BF16 tensors on Ascend950:
    ``a[M,K]``, ``b[K,N]``, ``out[M,N]``; M/N/K are multiples of 16 in
    [16, 256]. Output allocation and validation precede backend execution;
    inputs are never copied, cast or repacked. ``out`` must not overlap inputs.
    ``backend`` is ``cann``, ``ascendc``, ``triton`` or ``auto``. Auto consumes
    valid tuning records, then the configured deterministic ordering.
    Ascend C requires an installed AOT package and never falls back to JIT.
    This is synchronous eager execution, without a graph-capture promise.
    """
    import torch
    from .aclnn import enable_aclnn

    enable_aclnn()
    options = dict(backend=backend.value if isinstance(backend, Backend) else backend,
                   kernel_id=kernel_id)
    if out is None:
        return torch.ops.flashinfer.matmul(a, b, **options)
    torch.ops.flashinfer.matmul_out(a, b, out, **options)
    return out


def _matmul_impl(a, b, out=None, *, backend=None, kernel_id=None):
    import torch
    from dataclasses import replace
    from ._matmul import AOT_KERNEL_ID, OP, contract, output_metadata
    from .ascendc import tensor_contract_specialization
    from .jit.tensor_adapter import _reject_output_overlap

    # infershape output
    prospective = output_metadata(a, b, out)
    signature = build_signature(
        OP, (a, b) if out is None else (a, b, out),
        specializations={"tensor_contract": tensor_contract_specialization(
            AOT_KERNEL_ID, (a, b, prospective),
        )},
    )
    if out is None:
        signature = replace(
            signature, dtypes=signature.dtypes + (str(a.dtype).removeprefix("torch."),),
            shapes=signature.shapes + (prospective.shape,),
            layouts=signature.layouts + ("",),
            physical_layouts=tuple(
                signature.physical_layouts[i] if i < len(signature.physical_layouts) else ""
                for i in range(2)
            ) + ("ND",),
        )
    decision = dispatch(signature, current_hardware(), backend=backend, kernel_id=kernel_id)
    if out is None:
        out = torch.empty(prospective.shape, dtype=a.dtype, device=a.device)
    _reject_output_overlap(contract()[1].tensor_parameters, (a, b, out))
    _run_selected(decision.kernel_id, a, b, out)
    return out


def matmul_bias_silu(
    a, b, bias, out=None, *,
    backend: Union[str, Backend, None] = "auto", kernel_id: Optional[str] = None,
):
    """Compute ``silu(a @ b + bias)`` with the registered Ascend C kernel.

    All tensors are contiguous BF16 in NPU ND format: ``a[M,K]``, ``b[K,N]``,
    ``bias[N]`` and ``out[M,N]``. M, N and K must be multiples of 16 in
    [16, 256]. The dispatcher rejects unsupported shapes, dtypes and layouts.
    A verified installed package is used automatically; an absent package can
    compile only when JIT is enabled. This eager entry is not graph capturable.
    """
    import torch
    from .aclnn import enable_aclnn

    enable_aclnn()
    options = dict(backend=backend.value if isinstance(backend, Backend) else backend,
                   kernel_id=kernel_id)
    if out is None:
        return torch.ops.flashinfer.matmul_bias_silu(a, b, bias, **options)
    torch.ops.flashinfer.matmul_bias_silu_out(a, b, bias, out, **options)
    return out


def _matmul_bias_silu_impl(a, b, bias, out=None, *, backend="auto", kernel_id=None):
    """PrivateUse1 implementation shared by the functional and out schemas."""
    import torch
    from dataclasses import replace
    from types import SimpleNamespace
    from .ascendc import MATMUL_BIAS_SILU_KERNEL_ID, tensor_contract_specialization

    if out is None:
        # Validate the prospective output from metadata before allocating it;
        # unsupported huge shapes must reach dispatch without risking an OOM.
        m = a.shape[0] if a.ndim else 0
        n = b.shape[1] if b.ndim >= 2 else 0
        output_metadata = SimpleNamespace(
            shape=(m, n), dtype=torch.bfloat16, device=a.device,
            is_contiguous=lambda: True,
        )
    else:
        output_metadata = out
    tensors = (a, b, bias, output_metadata)
    signature = build_signature(
        "gemm.matmul_bias_silu", tensors if out is not None else tensors[:3],
        specializations={"tensor_contract": tensor_contract_specialization(
            MATMUL_BIAS_SILU_KERNEL_ID, tensors,
        )},
    )
    if out is None:
        signature = replace(
            signature, dtypes=signature.dtypes + ("bfloat16",),
            shapes=signature.shapes + (output_metadata.shape,),
            layouts=signature.layouts + ("",),
            physical_layouts=tuple(
                signature.physical_layouts[index]
                if index < len(signature.physical_layouts) else ""
                for index in range(3)
            ) + ("ND",),
        )
    decision = dispatch(signature, current_hardware(), backend=backend, kernel_id=kernel_id)
    if out is None:
        out = torch.empty(output_metadata.shape, dtype=torch.bfloat16, device=a.device)
    _run_selected(decision.kernel_id, a, b, bias, out)
    return out


def _run(op, tensors, specializations, args, backend, kernel_id):
    signature = build_signature(op, tensors, specializations=specializations)
    decision = dispatch(signature, current_hardware(), backend=backend, kernel_id=kernel_id)

    return _run_selected(decision.kernel_id, *args)


def mm_bf16(
    a, b, bias=None, pdl: bool = False, out=None, out_dtype=None,
    backend: Union[str, Backend, None] = "auto", *, kernel_id: Optional[str] = None,
):
    """bf16 矩阵乘（对标 flashinfer.gemm.mm_bf16）。

    ``a`` shape ``(m, k)`` row-major；``b`` shape ``(k, n)`` column-major；
    ``bias`` shape ``(n,)``，可选；返回 ``(m, n)``。

    Notes
    -----
    - ``pdl`` 是 NVIDIA Programmatic Dependent Launch 专属参数，昇腾无对应
      硬件能力，接受并忽略——与上游在非 CUDA 设备上 ``device_support_pdl``
      返回 False 的行为一致。
    - ``backend`` 的默认值由上游的 ``"cudnn"`` 改为 ``"auto"``（账本登记
      intentionally_different）：``"cudnn"`` 是 CUDA 专有后端名，在昇腾上
      没有对应物，保留该默认会让每次调用都失败。传入上游后端名会得到
      带迁移指引的报错。
    """
    import torch

    out_dtype = out_dtype if out_dtype is not None else torch.bfloat16
    del pdl  # 昇腾无 PDL；仅为签名级 drop-in 兼容保留
    return _run(
        _OP_MM_BF16, tuple(t for t in (a, b, bias) if t is not None), {},
        (a, b, bias, out, out_dtype), backend, kernel_id,
    )


def bmm_bf16(
    A, B, out=None, out_dtype=None,
    backend: Union[str, Backend, None] = "auto", *, kernel_id: Optional[str] = None,
):
    """批量 bf16 矩阵乘（对标 flashinfer.gemm.bmm_bf16）。

    ``A`` shape ``(b, m, k)``，``B`` shape ``(b, k, n)``，返回 ``(b, m, n)``。
    ``backend`` 默认值同 :func:`mm_bf16` 的说明。
    """
    import torch

    out_dtype = out_dtype if out_dtype is not None else torch.bfloat16
    return _run(_OP_BMM_BF16, (A, B), {}, (A, B, out, out_dtype), backend, kernel_id)


class SegmentGEMMWrapper:
    """分段 GEMM（对标 flashinfer.gemm.SegmentGEMMWrapper）。

    对一组按段切分的 token 各自乘以不同的权重矩阵：
    ``y[i] = x[i] @ W[weight_indices[i]]``。底层走 CANN 的
    ``npu_grouped_matmul``（group_type=0）。
    """

    def __init__(self, float_workspace_buffer=None, backend: str = "auto"):
        """``float_workspace_buffer`` 接受并忽略——昇腾侧 workspace 由算子
        自行管理，保留参数位以兼容上游构造方式。"""
        self._workspace = float_workspace_buffer
        self._backend = backend

    def reset_workspace_buffer(self, float_workspace_buffer=None, int_workspace_buffer=None) -> None:
        """重设 workspace（对标上游同名方法；昇腾侧仅记录，不参与计算）。"""
        self._workspace = float_workspace_buffer

    def run(
        self, x, weights, batch_size: int, weight_column_major: bool,
        out=None, seg_lens=None, seg_indptr=None, weight_indices=None,
        *, backend: Union[str, Backend, None] = None, kernel_id: Optional[str] = None,
    ):
        """执行分段 GEMM。

        ``seg_lens`` 与 ``seg_indptr`` 给其一即可：后者是前者的累加和并以 0 开头。
        ``x`` shape ``(sum(seg_lens), d_in)``；``weights`` shape
        ``(num_weights, d_in, d_out)``，``weight_column_major=True`` 时为
        ``(num_weights, d_out, d_in)``。返回 ``(sum(seg_lens), d_out)``。
        """
        import torch

        if seg_indptr is None:
            if seg_lens is None:
                raise ValueError("seg_lens 与 seg_indptr 必须给其一")
            zero = torch.zeros(1, dtype=torch.int64, device=seg_lens.device)
            seg_indptr = torch.cat([zero, seg_lens.to(torch.int64).cumsum(0)])
        tensors = tuple(
            t for t in (x, weights, seg_indptr, weight_indices) if t is not None
        )
        return _run(
            _OP_SEGMENT_GEMM, tensors,
            {"weight_column_major": bool(weight_column_major)},
            (x, weights, seg_indptr, weight_indices, bool(weight_column_major), out),
            backend if backend is not None else self._backend, kernel_id,
        )
