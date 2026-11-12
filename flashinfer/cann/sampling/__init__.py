"""CANN tensor filtering and native categorical sampling, loaded lazily."""

from flashinfer._sampling import descriptors
from flashinfer.runtime.schema import Backend


def sampling_descriptors():
    return descriptors(Backend.CANN)
