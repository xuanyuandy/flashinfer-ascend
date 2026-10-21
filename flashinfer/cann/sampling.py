"""CANN tensor filtering and native categorical sampling, loaded lazily."""

from .._sampling import descriptors
from ..runtime.schema import Backend


def sampling_descriptors():
    return descriptors(Backend.CANN)
