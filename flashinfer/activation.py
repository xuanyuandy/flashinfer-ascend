"""Upstream-compatible gated activation frontends."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .runtime import build_signature, current_hardware, dispatch, get_global_registry


def _run(name, input, out, enable_pdl):
    import torch

    # CUDA PDL is hardware-specific. Upstream also disables it on non-CUDA devices.
    del enable_pdl
    if input.ndim < 1 or input.shape[-1] % 2:
        raise ValueError("The last dimension must be even (2 * hidden_size).")
    expected = (*input.shape[:-1], input.shape[-1] // 2)
    if out is None:
        out = torch.empty(expected, dtype=input.dtype, device=input.device)
    if out.shape != expected or out.dtype != input.dtype or out.device != input.device:
        raise ValueError("out must have the expected shape, dtype and device")
    signature = build_signature("activation." + name, (input,), layouts=("ND",))
    decision = dispatch(signature, current_hardware(), backend=None)
    return get_global_registry().get(decision.kernel_id).runner(input, out)


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None) -> torch.Tensor:
    return _run("silu_and_mul", input, out, enable_pdl)


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None) -> torch.Tensor:
    return _run("gelu_and_mul", input, out, enable_pdl)


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None) -> torch.Tensor:
    return _run("gelu_tanh_and_mul", input, out, enable_pdl)


# Definitions and implementation choices are collected by the framework.
from flashinfer.operators.catalog import install_exports as _install_operator_exports
_install_operator_exports(globals(), domain='activation')
