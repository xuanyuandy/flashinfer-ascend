"""Decision cache isolation, invalidation and registry lifetime."""

from dataclasses import replace
import gc
import weakref

import pytest

from flashinfer.runtime import HardwareSignature, InputSignature
from flashinfer.runtime.dispatch import clear_dispatch_cache, dispatch
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import Backend, CapabilitySpec, KernelDescriptor


_SIGNATURE = InputSignature(op="test.dispatch_cache", dtypes=("float16",))
_HARDWARE = HardwareSignature(soc_model="Ascend910B", aicore_count=20)


def _descriptor(kernel_id, backend=Backend.CANN, priority=0):
    return KernelDescriptor(
        kernel_id=kernel_id,
        backend=backend,
        capability=CapabilitySpec(op=_SIGNATURE.op),
        priority=priority,
    )


def test_decision_cache_isolated_between_registries():
    ra = KernelRegistry([_descriptor("cann/probe/a")])
    rb = KernelRegistry([_descriptor("cann/probe/b")])
    assert ra.revision == rb.revision == 1

    # Keep both registries alive and never clear between dispatches. This also
    # runs directly, without pytest or conftest's cache-cleaning fixture.
    first_a = dispatch(_SIGNATURE, _HARDWARE, registry=ra)
    first_b = dispatch(_SIGNATURE, _HARDWARE, registry=rb)
    assert first_a.kernel_id == "cann/probe/a"
    assert first_b.kernel_id == "cann/probe/b"
    assert not first_a.cache_hit
    assert not first_b.cache_hit
    assert tuple(report.kernel_id for report in first_a.reports) == ("cann/probe/a",)
    assert tuple(report.kernel_id for report in first_b.reports) == ("cann/probe/b",)

    again_a = dispatch(_SIGNATURE, _HARDWARE, registry=ra)
    again_b = dispatch(_SIGNATURE, _HARDWARE, registry=rb)
    assert again_a == replace(first_a, cache_hit=True)
    assert again_b == replace(first_b, cache_hit=True)


@pytest.mark.parametrize("entry, backend, kernel_id", [
    ("register", Backend.CANN, "cann/probe/new"),
    ("register_external", Backend.EXTERNAL, "probe-external/probe/new"),
])
def test_registry_revision_invalidates_its_cached_decision(entry, backend, kernel_id):
    registry = KernelRegistry([_descriptor("cann/probe/base")])
    first = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    assert first.kernel_id == "cann/probe/base"
    assert not first.cache_hit
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit

    getattr(registry, entry)(_descriptor(kernel_id, backend, priority=10))
    assert registry.revision == 2
    updated = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    assert updated.kernel_id == kernel_id
    assert not updated.cache_hit
    assert updated.fallback_chain == (first.kernel_id,)
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry) == replace(
        updated, cache_hit=True,
    )


def test_cache_releases_registry_and_its_decisions():
    registry = KernelRegistry([_descriptor("cann/probe/transient")])
    decision = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
    registry_ref = weakref.ref(registry)
    decision_ref = weakref.ref(decision)
    assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit

    del registry, decision
    gc.collect()
    assert registry_ref() is None
    # Check that dead registries do not leave cache entries accumulating either.
    assert decision_ref() is None


def test_clear_dispatch_cache_clears_all_registries():
    registries = [
        KernelRegistry([_descriptor("cann/probe/a")]),
        KernelRegistry([_descriptor("cann/probe/b")]),
    ]
    for registry in registries:
        dispatch(_SIGNATURE, _HARDWARE, registry=registry)
        assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit

    clear_dispatch_cache()
    for registry in registries:
        decision = dispatch(_SIGNATURE, _HARDWARE, registry=registry)
        assert decision.kernel_id == registry.descriptors()[0].kernel_id
        assert not decision.cache_hit
        assert dispatch(_SIGNATURE, _HARDWARE, registry=registry).cache_hit
