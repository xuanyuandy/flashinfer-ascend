"""Torch/ACLNN integration. Import performs no registration or device work."""


def enable_aclnn():
    """Register production Torch schemas (idempotent, no execution-mode switch).

    Direct torch.ops callers register during application setup; public Python
    facades register automatically and delegate to these same Torch operators.
    Loading a kernel or native library remains lazy until real NPU execution.
    This does not declare graph capture/replay support.
    """
    from ._ops import enable

    enable()


__all__ = ["enable_aclnn"]
