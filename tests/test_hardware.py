"""Hardware discovery must use the same SoC family as kernel capabilities."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flashinfer.runtime.hardware import _normalize_soc_model, _probe


@pytest.mark.parametrize(
    "name,family",
    [
        ("Ascend950PR", "Ascend950"),
        ("Ascend950PR_9579", "Ascend950"),
        ("Ascend 950DT", "Ascend950"),
        ("Ascend950", "Ascend950"),
        ("Ascend910B3", "Ascend910B"),
        ("Ascend910C", "Ascend910C"),
        ("Ascend310P3", "Ascend310P"),
        ("FutureSoC", "FutureSoC"),
        ("", "unknown"),
    ],
)
def test_soc_family(name, family):
    assert _normalize_soc_model(name) == family


def test_a5_probe_preserves_core_count():
    npu = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda device: "Ascend950PR",
        get_device_properties=lambda device: SimpleNamespace(multi_processor_count=32),
    )
    with patch.dict(
        "sys.modules", torch=SimpleNamespace(npu=npu), torch_npu=SimpleNamespace()
    ):
        signature = _probe()
    assert signature.soc_model == "Ascend950"
    assert signature.aicore_count == 32
    assert signature.is_npu_available


def test_probe_without_npu():
    with patch.dict(
        "sys.modules",
        torch=SimpleNamespace(npu=SimpleNamespace(is_available=lambda: False)),
        torch_npu=SimpleNamespace(),
    ):
        signature = _probe()
    assert not signature.is_npu_available
