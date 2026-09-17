"""GEMM 系列：CANN fp8 GEMM 薄封装（torch_npu / aclnn 直通，§6.4）。

两接口均为 950 代际 fp8 特性，走 ``torch_npu.npu_grouped_matmul`` 的
fp8 输入路径（与 cann/moe.py 的 fp8 级联同源）：
- ``bmm_fp8``：batch 维视为分组（每组 m 行），per-tensor scale 在
  fp8 GEMM 之后统一折算；
- ``gemm_fp8_nt_groupwise``：K 方向按 gk 分块调用 fp8 GEMM，分组
  scale 按块折算累加（与 fp8_block_scale_moe 的 ``_fp8_block_gmm``
  同构）。全程 fp8 计算，无 bf16 中转。

950 真机待确认点（同 moe fp8 级联）：``npu_grouped_matmul`` 是否直接
接受 fp8 e4m3/e5m2 输入以及输出 dtype 推断规则；若 e5m2 不支持需从
``_SUPPORTED_DTYPES`` 剔除。
"""

from __future__ import annotations

from typing import Any, Tuple

from ..runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

# A/B 为 fp8（e4m3/e5m2），scale 为 fp32
_SUPPORTED_DTYPES = ("float8_e4m3fn", "float8_e5m2", "float32")
_SUPPORTED_SOCS = ("Ascend950",)


def _fp8_grouped_gemm(x8: Any, weight_list: list, group_list: list) -> Any:
    """fp8 grouped GEMM（950 上 npu_grouped_matmul 的 fp8 路径）。

    x8 (Σrows, K) fp8，weight_list 每组 (K, N)，group_list 为累计行数。
    与 cann/moe.py ``_fp8_gmm`` 同形态。
    """
    import torch
    import torch_npu

    y_list = torch_npu.npu_grouped_matmul(
        [x8],
        weight_list,
        group_list=group_list,
        split_item=1,
        group_type=0,
    )
    return torch.cat(list(y_list), dim=0)


def _bmm_fp8_cann(
    A: Any, B: Any, A_scale: Any, B_scale: Any, out: Any
) -> Any:
    """bmm_fp8 CANN 路径（950）：batch 作为分组走 fp8 grouped GEMM，
    per-tensor scale 在 GEMM 之后统一折算。"""
    b, m, k = A.shape
    n = B.shape[2]
    x = A.reshape(b * m, k).contiguous()
    weight_list = [B[i].contiguous() for i in range(b)]  # 每组 (k, n)
    group_list = [(i + 1) * m for i in range(b)]
    y = _fp8_grouped_gemm(x, weight_list, group_list).float().reshape(b, m, n)
    y = y * A_scale.float() * B_scale.float()
    out.copy_(y.to(out.dtype))
    return out


def _gemm_fp8_nt_groupwise_cann(
    a: Any,
    b: Any,
    a_scale: Any,
    b_scale: Any,
    scale_major_mode: str,
    scale_granularity_mnk: Any,
    out: Any,
) -> Any:
    """gemm_fp8_nt_groupwise CANN 路径（950）：K 方向按 gk 分块做
    fp8 GEMM，分组 scale 按块折算累加（fp32 累加器）。"""
    import torch

    gm, gn, gk = (int(x) for x in scale_granularity_mnk)
    m, k = a.shape
    n = b.shape[0]
    sa, sb = a_scale.float(), b_scale.float()
    if sa.shape != (m // gm, k // gk):
        sa = sa.t()  # MN-major 存放 → 转置归一（同 reference）
    if sb.shape != (n // gn, k // gk):
        sb = sb.t()
    del scale_major_mode  # 布局差异已在转置归一中吸收（同 reference）
    y = torch.zeros(m, n, dtype=torch.float32, device=a.device)
    for c in range(k // gk):
        sl = slice(c * gk, (c + 1) * gk)
        part = _fp8_grouped_gemm(
            a[:, sl].contiguous(),
            [b[:, sl].t().contiguous()],  # (gk, n)
            [m],
        ).float()  # (m, n)
        scale = sa[:, c].repeat_interleave(gm).unsqueeze(1) * sb[
            :, c
        ].repeat_interleave(gn).unsqueeze(0)
        y += part * scale
    out.copy_(y.to(out.dtype))
    return out


def gemm_descriptors() -> Tuple[KernelDescriptor, ...]:
    """CANN gemm 注册表。"""
    return (
        KernelDescriptor(
            kernel_id="cann/gemm/bmm_fp8_grouped_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="gemm.bmm_fp8",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_bmm_fp8_cann,
        ),
        KernelDescriptor(
            kernel_id="cann/gemm/gemm_fp8_nt_groupwise_block_v1",
            backend=Backend.CANN,
            compile_mode=CompileMode.NONE,
            capability=CapabilitySpec(
                op="gemm.gemm_fp8_nt_groupwise",
                dtypes=_SUPPORTED_DTYPES,
                soc_models=_SUPPORTED_SOCS,
            ),
            runner=_gemm_fp8_nt_groupwise_cann,
        ),
    )
