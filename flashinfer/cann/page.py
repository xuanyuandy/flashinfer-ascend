"""CANN ND paged scatter registration."""

from .._page import descriptors
from ..runtime.schema import Backend


def page_descriptors():
    return descriptors(Backend.CANN)
