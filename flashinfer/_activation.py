"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.reference.activation._family import (NAMES, execute, partial)


def descriptors(backend):
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend=backend, origin='flashinfer._activation')


def execute(*args, native, **kwargs):
    from importlib import import_module
    backend = "cann" if native else "reference"
    return getattr(import_module("flashinfer." + backend + ".activation._family"), "execute")(*args, **kwargs)
