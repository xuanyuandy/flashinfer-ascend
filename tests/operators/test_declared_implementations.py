"""Check the exported default route and each declared backend with independent CPU goldens."""

from contextlib import nullcontext
from importlib import import_module

import pytest

import flashinfer

from flashinfer.operators.catalog import cases_module, definition, entries
from flashinfer.operators.implementation import implementations
from flashinfer.operators.pytest_support import aot_params, implementation_param
from flashinfer.operators.testing import case_parameters, case_tolerances, numerical_cases
from flashinfer.runtime import Backend, CompileMode, KernelRegistry, get_global_registry, kernel_registry_scope


def _public_cases():
    for param in aot_params():
        impl, = param.values
        for index, configuration in enumerate(case_parameters(cases_module(impl.op), impl.kernel_id)):
            yield pytest.param(impl.op, configuration, marks=param.marks, id=f"{param.id}-case{index}")


@pytest.mark.parametrize("op,configuration", _public_cases())
def test_public_default_route(npu_runtime, aot_runtime, op, configuration):
    torch, torch_npu = npu_runtime
    operator = definition(op)
    if operator.public_entry:
        module, name = operator.public_entry.split(":")
        call = getattr(flashinfer, name, getattr(import_module(module), name))
    else:
        call = getattr(flashinfer, operator.name)
    cases = cases_module(op)
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = call(*args, **options)
    tolerances = case_tolerances(cases, configuration)
    if len(tolerances) == 1:
        actual, expected = (actual,), (expected,)
    assert len(actual) == len(expected) == len(tolerances)
    for output, golden, (atol, rtol) in zip(actual, expected, tolerances):
        if operator.schema:
            assert torch_npu.get_npu_format(output) == 2  # Typed public calls allocate ND outputs.
        if atol == rtol == 0:
            assert output.dtype == golden.dtype
        torch.testing.assert_close(output.cpu().double(), golden.double(), atol=atol, rtol=rtol)


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
