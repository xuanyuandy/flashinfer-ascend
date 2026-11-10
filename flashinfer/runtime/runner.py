"""Shared execution handoff after dispatch has selected a kernel identity."""

from .registry import get_global_registry
from .schema import DispatchError


def run_selected(kernel_id: str, *args, **kwargs):
    """Invoke the selected callable once, preserving arguments and exceptions."""
    descriptor = get_global_registry().get(kernel_id)
    if descriptor.runner is None:
        raise DispatchError(
            "kernel %s 已选中，但可执行产物尚未接入" % kernel_id
        )
    return descriptor.runner(*args, **kwargs)
