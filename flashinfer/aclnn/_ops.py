"""Framework Torch registration for declarations owned by logical operator modules."""

from functools import lru_cache

_registered = False


@lru_cache(maxsize=1)
def _declarations():
    from flashinfer.operators.runtime import torch_declarations
    return torch_declarations()


def enable():
    global _registered
    if _registered:
        return
    import torch
    from flashinfer.utils import _registration_lock
    from flashinfer._torch_library import _register_ascendc_ops, _validate_ascendc_ops

    if not (hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_fake")):
        raise RuntimeError("enable_aclnn requires torch >= 2.4")
    with _registration_lock:
        if _registered:
            return
        declarations = _declarations()
        _validate_ascendc_ops(declarations)
        _register_ascendc_ops(declarations)
        _registered = True
