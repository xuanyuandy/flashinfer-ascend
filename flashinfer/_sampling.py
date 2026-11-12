"""Compatibility exports; definitions and runners live in per-operator packages."""

from flashinfer.reference.sampling._family import (OP_NAMES, _chain, _draw, _matrix, _normalize, _parameter, _rng, _top_k, _top_p, execute, partial)


def descriptors(backend):
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend=backend, origin='flashinfer._sampling')


def execute(*args, native, **kwargs):
    from importlib import import_module
    backend = "cann" if native else "reference"
    return getattr(import_module("flashinfer." + backend + ".sampling._family"), "execute")(*args, **kwargs)
