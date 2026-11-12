"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.reference.page._family import (append_execute, partial)


def descriptors(backend):
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend=backend, origin='flashinfer._page')


def append_execute(*args, native, **kwargs):
    from importlib import import_module
    backend = "cann" if native else "reference"
    return getattr(import_module("flashinfer." + backend + ".page._family"), "append_execute")(*args, **kwargs)
