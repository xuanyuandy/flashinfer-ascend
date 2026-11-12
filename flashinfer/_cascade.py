"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.reference.cascade._family import (math, merge_states_execute, partial)


def descriptors(backend):
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend=backend, origin='flashinfer._cascade')


def merge_states_execute(*args, native, **kwargs):
    from importlib import import_module
    backend = "cann" if native else "reference"
    return getattr(import_module("flashinfer." + backend + ".cascade._family"), "merge_states_execute")(*args, **kwargs)
