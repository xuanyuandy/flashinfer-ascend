"""GEMM 系列：CANN fp8 GEMM 薄封装（torch_npu / aclnn 直通，§6.4）。

两接口均为 950 代际 fp8 特性，走 ``torch_npu.npu_grouped_matmul`` 的
fp8 输入路径（与 cann/moe.py 的 fp8 级联同源）：
- ``bmm_fp8``：batch 维视为分组（每组 m 行），per-tensor scale 在
  fp8 GEMM 之后统一折算；
- ``gemm_fp8_nt_groupwise``：K 方向按 gk 分块调用 fp8 GEMM，分组
  scale 按块折算累加（与 fp8_block_scale_moe 的 ``_fp8_block_gmm``
  同构）。全程 fp8 计算，无 bf16 中转。

FP8 使用 Tensor group_list 与堆叠的专家权重进入 ACLNN 新接口；单位
scale 保留原始 FP8 点积，显式输出 fp32 后再应用调用方的 scale。
"""

from __future__ import annotations

from typing import Any, Tuple

from flashinfer.runtime.schema import Backend, CapabilitySpec, CompileMode, KernelDescriptor

_SUPPORTED_DTYPES = ("float8_e4m3fn", "float8_e5m2", "float32")

_SUPPORTED_SOCS = ("Ascend950",)

_GENERAL_SOCS = ("Ascend910B", "Ascend950")

def _fp8_grouped_gemm(x8: Any, weight_list: list, group_list: list) -> Any:
    """fp8 grouped GEMM（950 上 npu_grouped_matmul 的 fp8 路径）。

    x8 (Σrows, K) fp8，weight_list 每组 (K, N)，group_list 为累计行数。
    与 cann/moe.py ``_fp8_gmm`` 同形态。
    """
    import torch
    import torch_npu

    # List[int] group_list 会走旧接口，不支持 A5 FP8。量化接口要求显式
    # scale；fp32 输出避免原始点积在外部缩放前溢出或再次量化到 FP8。
    # CANN Stack 尚不支持 FP8；以等宽 uint8 搬运，保留每个 FP8 位模式。
    weights = torch.stack([w.view(torch.uint8) for w in weight_list]).view(
        weight_list[0].dtype
    )
    groups = torch.tensor(group_list, dtype=torch.int64, device=x8.device)
    y_list = torch_npu.npu_grouped_matmul(
        [x8],
        [weights],
        scale=[
            torch.ones(
                len(weight_list), weights.shape[-1], dtype=torch.float32,
                device=x8.device,
            )
        ],
        per_token_scale=[torch.ones(x8.shape[0], dtype=torch.float32, device=x8.device)],
        group_list=groups,
        split_item=3,
        group_type=0,
        output_dtype=torch.float32,
    )
    return y_list[0]
