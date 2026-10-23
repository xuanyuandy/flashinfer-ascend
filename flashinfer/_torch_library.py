"""T8 registrations, created only by the explicit setup entry point.

Both schemas return void: output metadata lives in the caller-owned tensors.
The public RMSNorm frontend creates its output with empty_like; under tracing
that creates only a FakeTensor/meta tensor. The fake kernels therefore need no
tensor operations at all, including allocation or data/shape-dependent work.
"""

from functools import lru_cache
from types import SimpleNamespace
from typing import Optional


def _rmsnorm_meta(out, input, weight, eps, backend, kernel_id) -> None:
    """Only out is written; its shape/dtype/strides/device are already known."""
    return None


def _fused_add_rmsnorm_meta(input, residual, weight, eps, backend, kernel_id) -> None:
    """input and residual keep their metadata; this op has no return tensors."""
    return None


@lru_cache(maxsize=1)
def _register_norm_ops():
    # Keep even a direct import of this module free of optional dependencies.
    import torch

    from . import norm
    from .utils import register_custom_op, register_fake_op

    @register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
    def rmsnorm(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        backend: Optional[str],
        kernel_id: Optional[str],
    ) -> None:
        norm._rmsnorm_impl(out, input, weight, eps, backend, kernel_id)

    register_fake_op("flashinfer::rmsnorm", _rmsnorm_meta)

    @register_custom_op(
        "flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"),
    )
    def fused_add_rmsnorm(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        backend: Optional[str],
        kernel_id: Optional[str],
    ) -> None:
        norm._fused_add_rmsnorm_impl(input, residual, weight, eps, backend, kernel_id)

    register_fake_op("flashinfer::fused_add_rmsnorm", _fused_add_rmsnorm_meta)
    return SimpleNamespace(rmsnorm=rmsnorm, fused_add_rmsnorm=fused_add_rmsnorm)
