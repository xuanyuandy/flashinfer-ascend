"""Host fixtures need no NPU/Torch by default; device setup is an explicit opt-in."""

from __future__ import annotations

import os
import pytest

from flashinfer.runtime.dispatch import clear_dispatch_cache
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.registry import KernelRegistry
from flashinfer.runtime.schema import (
    Backend,
    CapabilitySpec,
    CompileMode,
    KernelDescriptor,
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
