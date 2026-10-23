"""T8/T14 registrations, created only by the explicit setup entry point.

The norm schemas return void: output metadata lives in caller-owned tensors.
The public RMSNorm frontend creates its output with empty_like; under tracing
that creates only a FakeTensor/meta tensor. Their fake kernels need no
tensor operations at all, including allocation or data/shape-dependent work.
"""

from functools import lru_cache
from types import SimpleNamespace
from typing import Optional

_ascendc_ops = {}


def _validate_ascendc_ops(declarations):
    """Preflight the entire setup batch before changing the dispatcher."""
    if not declarations:
        return
    import torch

    from .jit.torch_library import AscendCOp

    seen = set()
    existing = set(torch._C._dispatch_get_all_op_names())
    for declaration in declarations:
        if not isinstance(declaration, AscendCOp):
            raise TypeError("ascendc_ops must contain AscendCOp declarations")
        name = declaration.name
        if name in seen:
            raise ValueError("duplicate Ascend C op in setup batch: " + name)
        seen.add(name)
        previous = _ascendc_ops.get(name)
        if previous is not None:
            if previous[0] is not declaration:
                raise ValueError("Ascend C op already enabled; use a fresh process: " + name)
        elif name in existing:
            raise ValueError("op already registered: " + name)
        schema = torch._C.parse_schema(name + declaration.schema)
        mutated = {arg.name for arg in schema.arguments
                   if arg.alias_info is not None and arg.alias_info.is_write}
        if mutated != set(declaration.mutates_args):
            raise ValueError("schema mutation annotations do not match mutates_args: " + name)
        if not any("Tensor" in str(arg.type) for arg in schema.arguments):
            raise ValueError("Ascend C ops require a tensor argument: " + name)
        if any(result.alias_info is not None for result in schema.returns):
            raise ValueError("Ascend C custom ops cannot return aliases: " + name)


def _register_ascendc_ops(declarations):
    if not declarations:
        return
    import torch

    from .utils import register_custom_op, register_fake_op

    # Works before torch_npu is imported and after PrivateUse1 is renamed npu.
    device_type = torch._C._get_privateuse1_backend_name()
    for declaration in declarations:
        if declaration.name in _ascendc_ops:
            continue
        op = register_custom_op(
            declaration.name, declaration.device_impl,
            schema=declaration.schema, mutates_args=declaration.mutates_args,
            device_types=device_type,
        )
        register_fake_op(declaration.name, declaration.meta_impl)
        # Keep the CustomOpDef and its library alive for the process lifetime.
        _ascendc_ops[declaration.name] = declaration, op


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
