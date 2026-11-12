"""CANN ND paged scatter registration."""

from flashinfer._page import descriptors
from flashinfer.runtime.schema import Backend


def page_descriptors():
    return descriptors(Backend.CANN)
