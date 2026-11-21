"""Host fixtures need no NPU/Torch by default; device setup is an explicit opt-in."""

from __future__ import annotations

import os
import pytest

from flashinfer.runtime.dispatch import clear_dispatch_cache
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.registry import KernelRegistry, _reset_global_registry_for_testing
from flashinfer.runtime.targets import GENERAL_ASCEND_SOCS, canonical_soc_model
from flashinfer.runtime.schema import (
    Backend,
    CapabilitySpec,
    CompileMode,
    KernelDescriptor,
)


def _soc_scope(value):
    aliases = {"a2": "Ascend910B", "a3": "Ascend910C", "a5": "Ascend950",
               "all": "all", "auto": "auto"}
    if value.lower() in aliases:
        return aliases[value.lower()]
    return canonical_soc_model(value)


def pytest_addoption(parser):
    parser.addoption(
        "--soc", type=_soc_scope, metavar="auto|all|A2|A3|A5|SOC",
        help="Run tests for the local chip by default. Collection alone defaults to all target SoCs.",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    requirements = {}
    for item in items:
        for marker in item.iter_markers("requires_soc"):
            if not marker.args or not set(marker.args).issubset(GENERAL_ASCEND_SOCS):
                raise pytest.UsageError(f"{item.nodeid}: requires_soc needs explicit canonical SoC names")
            requirements.setdefault(item, set(GENERAL_ASCEND_SOCS)).intersection_update(marker.args)
    target = config.getoption("--soc")
    if target is None:
        if config.getoption("collectonly") or not requirements:
            return
        target = "auto"
    if target == "all":
        return
    if target == "auto":
        from flashinfer.runtime import current_hardware
        target = current_hardware().soc_model
        if target not in GENERAL_ASCEND_SOCS:
            raise pytest.UsageError("--soc=auto requires a visible supported NPU; use --soc=all for host-only tests")
    selected, incompatible = [], []
    for item in items:
        (incompatible if item in requirements and target not in requirements[item] else selected).append(item)
    if incompatible:
        config.hook.pytest_deselected(items=incompatible)
    items[:] = selected
    config._flashinfer_soc_scope = (target, len(incompatible))


def pytest_collection_finish(session):
    target = session.config.getoption("--soc")
    if target in (None, "auto", "all") or session.config.getoption("collectonly"):
        return
    from flashinfer.runtime import current_hardware
    actual = current_hardware().soc_model
    if actual != target:
        raise pytest.UsageError(f"--soc selects {target}; local device is {actual}. "
                                "Use --collect-only to inspect another chip's cases.")


def pytest_terminal_summary(terminalreporter, config):
    scope = getattr(config, "_flashinfer_soc_scope", None)
    if scope is not None:
        target, count = scope
        terminalreporter.write_line(
            f"SoC scope: {target}; {count} incompatible-chip tests deselected (not verified)."
        )


def pytest_sessionstart(session):
    """Opt in to a free test device without hiding any visible devices."""
    value = os.environ.get("FLASHINFER_TEST_NPU_DEVICE")
    if value is not None:
        import torch
        import torch_npu  # noqa: F401

        index = int(value)
        if not 0 <= index < torch.npu.device_count():
            raise ValueError("FLASHINFER_TEST_NPU_DEVICE is outside the visible device range")
        torch.npu.set_device(index)


@pytest.fixture
def torch_for_class(request):
    import torch

    request.instance.torch = torch


@pytest.fixture(scope="module")
def npu_device():
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.fail("selected test requires an available NPU")
    return torch.device("npu", torch.npu.current_device())


@pytest.fixture(scope="module")
def a5_device(npu_device):
    from flashinfer.runtime import current_hardware

    if current_hardware().soc_model != "Ascend950":
        pytest.fail("selected test requires an Ascend950 device")
    return npu_device


@pytest.fixture
def cann_a5_device(a5_device, monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "cann")
    return a5_device


@pytest.fixture
def clean_global_registry():
    yield
    _reset_global_registry_for_testing()


@pytest.fixture(autouse=True)
def _clean_dispatch_cache():
    clear_dispatch_cache()
    yield
    clear_dispatch_cache()


@pytest.fixture
def npu_hardware() -> HardwareSignature:
    return HardwareSignature(soc_model="Ascend910B", aicore_count=20)


@pytest.fixture
def decode_registry() -> KernelRegistry:
    """attention.decode 的四个内置 kernel：覆盖启发式阶梯的全部层级。"""
    return KernelRegistry(
        [
            KernelDescriptor(
                kernel_id="ascendc/attention/decode_v1",
                backend=Backend.ASCENDC,
                capability=CapabilitySpec(
                    op="attention.decode", dtypes=("float16", "bfloat16")
                ),
                compile_mode=CompileMode.AOT,
                priority=1,
            ),
            KernelDescriptor(
                kernel_id="cann/attention/decode_aclnn_v1",
                backend=Backend.CANN,
                capability=CapabilitySpec(op="attention.decode"),
                compile_mode=CompileMode.NONE,
            ),
            KernelDescriptor(
                kernel_id="triton/attention/decode_v1",
                backend=Backend.TRITON,
                capability=CapabilitySpec(op="attention.decode"),
                compile_mode=CompileMode.JIT,
            ),
            KernelDescriptor(
                kernel_id="reference/attention/decode_v1",
                backend=Backend.REFERENCE,
                capability=CapabilitySpec(op="attention.decode"),
                compile_mode=CompileMode.NONE,
                runner=lambda *args, **kwargs: None,
            ),
        ]
    )
