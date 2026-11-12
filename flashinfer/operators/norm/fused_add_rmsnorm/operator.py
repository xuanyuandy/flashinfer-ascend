"""Backend-independent declaration for norm.fused_add_rmsnorm."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='fused_add_rmsnorm', domain='norm', op='norm.fused_add_rmsnorm', public_entry='flashinfer.norm:fused_add_rmsnorm')


from functools import lru_cache
from typing import Optional


def _fused_add_rmsnorm_meta(input, residual, weight, eps, backend, kernel_id) -> None:
    """input and residual keep their metadata; this op has no return tensors."""
    return None


@lru_cache(maxsize=1)
def register_torch_op():
    import torch
    from flashinfer import norm
    from flashinfer.utils import register_custom_op, register_fake_op

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
    return fused_add_rmsnorm
