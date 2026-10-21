"""Native CANN SwiGLU/GeluMul descriptors."""

from .._activation import descriptors
from ..runtime.schema import Backend


def activation_descriptors():
    return descriptors(Backend.CANN)
