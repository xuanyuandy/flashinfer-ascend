"""norm 对外 API 门面（对标 flashinfer.norm，§2.3、§3.1）。

函数名、参数名、默认值与语义对齐上游（flashinfer-ai/flashinfer
flashinfer/norm.py）。门面只做三件事：构建输入签名 → 委托 dispatcher
选择 kernel → 拉起执行。一行 kernel 代码都没有（§2.3）。

与上游的显式差异（§3.1 规则 2，parity 清单标记 compatible）：
``enable_pdl`` 是 NVIDIA Programmatic Dependent Launch 专属参数，
NPU 无对应机制——为保持 drop-in 签名兼容予以接受并忽略，不影响语义。
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

_OP_RMSNORM = "norm.rmsnorm"
_OP_FUSED_ADD_RMSNORM = "norm.fused_add_rmsnorm"
_OP_GEMMA_RMSNORM = "norm.gemma_rmsnorm"
_OP_GEMMA_FUSED_ADD_RMSNORM = "norm.gemma_fused_add_rmsnorm"
_OP_LAYERNORM = "norm.layernorm"


def _run_selected(kernel_id: str, *args, **kwargs):
    """拉起 dispatch 选中的 kernel。产物未接入时显式报错。"""
    descriptor = get_global_registry().get(kernel_id)
    if descriptor.runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入"
            "（AOT 产物加载 / JIT 编译拉起随后续里程碑接入，§15）" % kernel_id
        )
    return descriptor.runner(*args, **kwargs)


def rmsnorm(
    input,
    weight,
    eps: float = 1e-6,
    out=None,
    enable_pdl: Optional[bool] = None,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``，最后一维为归一化维。

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape (batch_size, hidden_size)。
    weight : torch.Tensor
        Weight tensor, shape (hidden_size,)。
    eps : float
        Epsilon for numerical stability。
    out : Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor
        inplace。
    enable_pdl : Optional[bool]
        NVIDIA PDL 专属参数，NPU 无对应机制，接受并忽略（见模块 docstring）。
    backend / kernel_id
        调度覆盖参数（§3.2.2，与计算语义正交）。

    Returns
    -------
    output : torch.Tensor
        Normalized tensor, shape 与 input 相同。
    """
    del enable_pdl  # 上游在非 CUDA 设备上同样关闭 PDL，语义兼容
    import torch  # 延迟导入：模块本身不依赖 torch（§14.1 host 单测）

    if out is None:
        out = torch.empty_like(input)
    signature = build_signature(
        _OP_RMSNORM,
        (input, weight),
        specializations={"eps": float(eps)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(decision.kernel_id, out, input, weight, float(eps))
    return out


def fused_add_rmsnorm(
    input,
    residual,
    weight,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
) -> None:
    """Fused add root mean square normalization（原地修改 input 与 residual）。

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape (batch_size, hidden_size)。
    residual : torch.Tensor
        Residual tensor, shape (batch_size, hidden_size)。
    weight : torch.Tensor
        Weight tensor, shape (hidden_size,)。
    eps : float
        Epsilon for numerical stability。
    enable_pdl : Optional[bool]
        NVIDIA PDL 专属参数，NPU 无对应机制，接受并忽略（见模块 docstring）。
    backend / kernel_id
        调度覆盖参数（§3.2.2，与计算语义正交）。
    """
    del enable_pdl  # 上游在非 CUDA 设备上同样关闭 PDL，语义兼容
    signature = build_signature(
        _OP_FUSED_ADD_RMSNORM,
        (input, residual, weight),
        specializations={"eps": float(eps)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(decision.kernel_id, input, residual, weight, float(eps))


def gemma_rmsnorm(
    input,
    weight,
    eps: float = 1e-6,
    out=None,
    enable_pdl: Optional[bool] = None,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """Gemma RMSNorm：``out = (input / RMS(input)) * (weight + 1)``。

    对标上游 ``flashinfer.norm.gemma_rmsnorm``——与 ``rmsnorm`` 的唯一
    差异是缩放因子为 ``weight + 1``（Gemma 约定）。``enable_pdl`` 为
    NVIDIA PDL 专属参数，NPU 无对应机制，接受并忽略。
    """
    del enable_pdl  # 上游在非 CUDA 设备上同样关闭 PDL，语义兼容
    import torch

    if out is None:
        out = torch.empty_like(input)
    signature = build_signature(
        _OP_GEMMA_RMSNORM,
        (input, weight),
        specializations={"eps": float(eps)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(decision.kernel_id, out, input, weight, float(eps))
    return out


def gemma_fused_add_rmsnorm(
    input,
    residual,
    weight,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
) -> None:
    """Gemma 融合加残差 RMSNorm（原地语义，无返回值），对标上游同名接口。

    Step 1: ``residual[i] += input[i]``（写回 ``residual``）
    Step 2: ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``
    （写回 ``input``）

    CANN 无该算子（``npu_add_rms_norm`` 缩放因子为 ``gamma`` 而非
    ``gamma + 1``），默认走 Triton 自定义 kernel。
    """
    del enable_pdl  # 上游在非 CUDA 设备上同样关闭 PDL，语义兼容
    signature = build_signature(
        _OP_GEMMA_FUSED_ADD_RMSNORM,
        (input, residual, weight),
        specializations={"eps": float(eps)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(decision.kernel_id, input, residual, weight, float(eps))


def layernorm(
    input,
    gemma,
    beta,
    eps: float = 1e-6,
    *,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """LayerNorm：``out = (input - E[input]) / sqrt(Var[input] + eps) * gemma + beta``。

    对标上游 ``flashinfer.norm.layernorm``（v0.6.0 起；参数名 gemma/beta
    与上游保持一致，即通常的 gamma/beta）。返回形状与 ``input`` 相同、
    dtype 与 ``input`` 相同的新张量。
    """
    import torch

    out = torch.empty_like(input)
    signature = build_signature(
        _OP_LAYERNORM,
        (input, gemma, beta),
        specializations={"eps": float(eps)},
    )
    decision = dispatch(
        signature, current_hardware(), backend=backend, kernel_id=kernel_id
    )
    _run_selected(decision.kernel_id, out, input, gemma, beta, float(eps))
    return out
