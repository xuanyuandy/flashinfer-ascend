"""GEMM 系列对外 API（§3.1 薄门面层）。

对齐上游 flashinfer.gemm 的 fp8 GEMM 接口：
- ``bmm_fp8``：fp8 per-tensor scale 批量矩阵乘；
- ``gemm_fp8_nt_groupwise``：fp8 分组 scale NT 布局矩阵乘。

硬件支持说明（§12 验收口径）：
fp8 矩阵乘在 CANN 中仅 950 代际（Atlas A3）具备原生算子；910B 上
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
    get_global_registry,
)
from .runtime.schema import DispatchError

_OP_BMM_FP8 = "gemm.bmm_fp8"
_OP_GEMM_FP8_NT_GROUPWISE = "gemm.gemm_fp8_nt_groupwise"


def _run_selected(kernel_id: str, *args, **kwargs):
    """拉起 dispatch 选中的 kernel。产物未接入时显式报错。"""
    descriptor = get_global_registry().get(kernel_id)
    if descriptor.runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入"
            "（AOT 产物加载 / JIT 编译拉起随后续里程碑接入，§15）" % kernel_id
        )
    return descriptor.runner(*args, **kwargs)


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
