"""Run shared operator goldens for each declared non-native backend with cases."""

from contextlib import nullcontext

import pytest

from flashinfer.operators.catalog import entries
from flashinfer.operators.implementation import implementations
from flashinfer.operators.pytest_support import implementation_param
from flashinfer.operators.testing import numerical_cases
from flashinfer.runtime import Backend, CompileMode, KernelRegistry, get_global_registry, kernel_registry_scope


def _cases():
    with_samples = {entry["op"] for entry in entries() if entry["cases"]}
    return [implementation_param(impl) for impl in implementations(scope=None)
            if impl.backend != Backend.ASCENDC and impl.op in with_samples]


@pytest.mark.parametrize("impl", _cases())
def test_numerical_cases_by_kernel_id(npu_runtime, impl, monkeypatch):
    torch, torch_npu = npu_runtime
    descriptor = impl.bind()
    if descriptor.compile_mode == CompileMode.JIT:
        monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    scope = nullcontext()
    if impl.scope == "experimental":
        registry = KernelRegistry((*get_global_registry().descriptors(), descriptor))
        registry.freeze()
        scope = kernel_registry_scope(registry)
    with scope:
        numerical_cases(torch, torch_npu, impl.op, kernel_id=impl.kernel_id, backend=impl.backend.value)
