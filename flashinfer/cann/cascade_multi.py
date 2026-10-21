"""TensorList attention update for stacked FlashInfer states."""

from .._cascade import descriptors
from ..runtime.schema import Backend


def cascade_multi_descriptors():
    return descriptors(Backend.CANN)
