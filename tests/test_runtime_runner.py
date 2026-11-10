"""The execution handoff preserves scope, caller objects and failure boundaries."""

import pytest

from flashinfer.runtime import DispatchError, KernelDescriptor, KernelRegistry
from flashinfer.runtime.registry import kernel_registry_scope
from flashinfer.runtime.runner import run_selected


def registry_with(*descriptors):
    registry = KernelRegistry(descriptors)
    registry.freeze()
    return registry


def test_execution_uses_scoped_registry_and_original_arguments():
    argument, output, result = object(), object(), object()
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    outer = registry_with(KernelDescriptor("runner/probe", op="probe", runner=execute))
    inner = registry_with(KernelDescriptor("runner/probe", op="probe", runner=lambda *a, **k: output))
    with kernel_registry_scope(outer):
        with kernel_registry_scope(inner):
            assert run_selected("runner/probe", argument, out=output) is output
        assert run_selected("runner/probe", argument, out=output) is result
    assert calls == [((argument,), {"out": output})]


def test_missing_id_and_missing_callable_fail_explicitly():
    registry = registry_with(KernelDescriptor("runner/unbound", op="probe"))
    with kernel_registry_scope(registry):
        with pytest.raises(KeyError, match="runner/unknown"):
            run_selected("runner/unknown")
        with pytest.raises(DispatchError, match="runner/unbound.*可执行产物尚未接入"):
            run_selected("runner/unbound")


def test_runner_failure_propagates_without_trying_another_implementation():
    failure = RuntimeError("execution failed")
    other_calls = []

    def fail():
        raise failure

    registry = registry_with(
        KernelDescriptor("runner/fail", op="probe", runner=fail),
        KernelDescriptor("runner/other", op="probe", runner=lambda: other_calls.append(True)),
    )
    with kernel_registry_scope(registry), pytest.raises(RuntimeError) as caught:
        run_selected("runner/fail")
    assert caught.value is failure
    assert other_calls == []
