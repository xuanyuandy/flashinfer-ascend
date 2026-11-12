"""Native CANN SwiGLU/GeluMul descriptors."""

from flashinfer._activation import descriptors
from flashinfer.runtime.schema import Backend


def activation_descriptors():
    return descriptors(Backend.CANN)
