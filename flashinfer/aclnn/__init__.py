"""Torch/ACLNN integration. Import performs no registration or device work."""


def enable_aclnn():
    """Register production Torch schemas and require the native ACLNN adapter.

    Call once during application setup, before tracing/concurrent execution.
    Loading a kernel or native library remains lazy until real NPU execution.
    This does not declare graph capture/replay support.
    """
    from ._ops import enable

    enable()


__all__ = ["enable_aclnn"]
