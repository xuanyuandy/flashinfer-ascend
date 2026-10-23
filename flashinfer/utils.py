"""Upstream-compatible registration helpers with an explicit graph-mode opt-in.

Importing this module does not import torch or any device/compiler package.
Unlike upstream's current identity decorators, the two helpers really register
when called. Builtin registration happens only in ``enable_torch_library``;
ordinary eager calls retain their existing dispatch path by default.
"""

from threading import Lock
from typing import Callable, Iterable, Optional, Sequence, Union

_registration_lock = Lock()


def register_custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: Optional[Union[str, Sequence[str]]] = None,
    schema: Optional[str] = None,
) -> Callable:
    """Register a real custom op, using the upstream helper's exact signature.

    Both decorator and direct-call forms follow ``torch.library.custom_op``.
    Registration requires torch >= 2.4 and never silently becomes a no-op.
    """
    import torch

    if not hasattr(torch.library, "custom_op"):
        raise RuntimeError("register_custom_op requires torch >= 2.4")
    return torch.library.custom_op(
        name, fn, mutates_args=mutates_args,
        device_types=device_types, schema=schema,
    )


def register_fake_op(
    name: str,
    fn: Optional[Callable] = None,
) -> Callable:
    """Register metadata-only behavior with the upstream helper's signature."""
    import torch

    if not hasattr(torch.library, "register_fake"):
        raise RuntimeError("register_fake_op requires torch >= 2.4")
    return torch.library.register_fake(name, fn)


def enable_torch_library(*, ascendc_ops=()) -> None:
    """Enable registered ops for this process, before torch.compile.

    This is an explicit, idempotent opt-in. By default the public frontends use
    their original eager path, matching upstream's choice to avoid registration
    overhead. Call this during application setup, outside tracing/capture and
    before concurrent execution. It registers ``flashinfer::rmsnorm`` and
    ``flashinfer::fused_add_rmsnorm`` and routes their frontends through them.

    The registered implementations still perform the original runtime dispatch;
    backend/kernel overrides and numerical behavior are unchanged. Registration
    does not probe hardware, run kernels, or declare aclgraph capture safety.

    ``ascendc_ops`` contains explicit ``jit.torch_library.AscendCOp``
    declarations backed by already prepared implementations. They register in
    ``flashinfer::`` with PrivateUse1 device kernels and separate fake kernels.
    Setup never builds or loads an artifact. Reusing a declaration is idempotent;
    replacing an enabled implementation requires a fresh process.
    """
    import torch

    # Check on every setup call, including one following a previous enable.
    if not (hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_fake")):
        raise RuntimeError("enable_torch_library requires torch >= 2.4")
    from . import norm
    from ._torch_library import _register_ascendc_ops, _validate_ascendc_ops

    with _registration_lock:
        declarations = tuple(ascendc_ops)
        _validate_ascendc_ops(declarations)
        if norm._torch_library_ops is None:
            from ._torch_library import _register_norm_ops

            norm._torch_library_ops = _register_norm_ops()
        _register_ascendc_ops(declarations)
