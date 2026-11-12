"""Backend-independent declaration for norm.rmsnorm."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='rmsnorm', domain='norm', op='norm.rmsnorm', public_entry='flashinfer.norm:rmsnorm')


from functools import lru_cache
from typing import Optional


def _rmsnorm_meta(out, input, weight, eps, backend, kernel_id) -> None:
    """Only out is written; its shape/dtype/strides/device are already known."""
    return None


@lru_cache(maxsize=1)
def register_torch_op():
    import torch
    from flashinfer import norm
    from flashinfer.utils import register_custom_op, register_fake_op

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
    return rmsnorm
