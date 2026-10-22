"""Hardware discovery must use the same SoC family as kernel capabilities."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flashinfer.runtime.hardware import (
    HardwareSignature, _normalize_soc_model, _probe, _probe_ub_size_bytes,
    _reset_hardware_cache_for_testing, current_hardware,
)


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
    query = Mock(return_value=(221184, 0))
    with patch.dict("sys.modules", torch=SimpleNamespace(npu=npu),
                    torch_npu=SimpleNamespace(),
                    acl=SimpleNamespace(rt=SimpleNamespace(get_device_info=query))):
        signature = _probe()
    assert signature.soc_model == "Ascend950"
    assert signature.aicore_count == 32
    assert signature.is_npu_available
    assert signature.ub_size_bytes == 221184
    query.assert_called_once_with(0, 204)


def test_probe_without_npu():
    with patch.dict(
        "sys.modules",
        torch=SimpleNamespace(npu=SimpleNamespace(is_available=lambda: False)),
        torch_npu=SimpleNamespace(),
    ), patch("flashinfer.runtime.hardware._probe_ub_size_bytes") as query:
        signature = _probe()
    assert not signature.is_npu_available
    assert signature.ub_size_bytes == 0
    query.assert_not_called()


def test_ub_default_and_frozen_hardware_identity():
    unknown = HardwareSignature()
    known = HardwareSignature("Ascend950", 56, 221184)
    assert unknown.ub_size_bytes == 0
    assert known == HardwareSignature("Ascend950", 56, 221184)
    assert hash(known) == hash(HardwareSignature("Ascend950", 56, 221184))
    assert len({known, replace(known, ub_size_bytes=0), replace(known, ub_size_bytes=262144)}) == 3
    with pytest.raises(FrozenInstanceError):
        known.ub_size_bytes = 0


@pytest.mark.parametrize("size", [-1, None, True, 216.0, "221184"])
def test_invalid_ub_capacity_is_rejected(size):
    with pytest.raises(ValueError, match="ub_size_bytes"):
        HardwareSignature(ub_size_bytes=size)


@pytest.mark.parametrize("response", [
    (0, 0), (-1, 0), (221184, 507899), (None, 0), (True, 0), ("221184", 0), None,
])
def test_unknown_or_failed_acl_capacity_is_zero(response):
    with patch.dict("sys.modules", acl=SimpleNamespace(
        rt=SimpleNamespace(get_device_info=lambda *args: response),
    )):
        assert _probe_ub_size_bytes(0) == 0


@pytest.mark.parametrize("acl_module", [
    None, SimpleNamespace(),
    SimpleNamespace(rt=SimpleNamespace(get_device_info=Mock(side_effect=RuntimeError("unsupported")))),
])
def test_acl_failure_preserves_detected_soc_and_cores(acl_module):
    npu = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda device: "Ascend950PR_9579",
        get_device_properties=lambda device: SimpleNamespace(multi_processor_count=56),
    )
    with patch.dict("sys.modules", torch=SimpleNamespace(npu=npu),
                    torch_npu=SimpleNamespace(), acl=acl_module):
        assert _probe() == HardwareSignature("Ascend950", 56, 0)


def test_device_properties_failure_does_not_lose_ub_capacity():
    npu = SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda device: "Ascend950PR",
        get_device_properties=Mock(side_effect=RuntimeError("no properties")),
    )
    with patch.dict("sys.modules", torch=SimpleNamespace(npu=npu),
                    torch_npu=SimpleNamespace(), acl=SimpleNamespace(
                        rt=SimpleNamespace(get_device_info=lambda *args: (221184, 0)),
                    )):
        assert _probe() == HardwareSignature("Ascend950", 0, 221184)


@pytest.mark.parametrize("modules", [
    {"torch": None}, {"torch_npu": None},
    {"torch": SimpleNamespace(npu=SimpleNamespace(is_available=Mock(side_effect=RuntimeError("no driver"))))},
])
def test_host_probe_without_runtime_or_driver_still_dispatches(modules, decode_registry):
    from flashinfer.runtime.dispatch import dispatch
    from flashinfer.runtime.signature import InputSignature

    with patch.dict("sys.modules", modules):
        hardware = _probe()
    assert hardware == HardwareSignature()
    assert dispatch(InputSignature("attention.decode"), hardware, registry=decode_registry).kernel_id


def test_hardware_cache_retains_ub_without_reprobing():
    signature = HardwareSignature("Ascend950", 56, 221184)
    _reset_hardware_cache_for_testing()
    try:
        with patch("flashinfer.runtime.hardware._probe", return_value=signature) as probe:
            assert current_hardware() is signature
            assert current_hardware() is signature
            probe.assert_called_once_with()
    finally:
        _reset_hardware_cache_for_testing()


def test_ub_capacity_partitions_dispatch_cache(decode_registry):
    from flashinfer.runtime.dispatch import dispatch
    from flashinfer.runtime.signature import InputSignature

    signature = InputSignature("attention.decode")
    hardware = HardwareSignature("Ascend950", 56, 221184)
    first = dispatch(signature, hardware, registry=decode_registry)
    unknown = dispatch(signature, replace(hardware, ub_size_bytes=0), registry=decode_registry)
    different = dispatch(signature, replace(hardware, ub_size_bytes=262144), registry=decode_registry)
    again = dispatch(signature, replace(hardware), registry=decode_registry)
    assert first.kernel_id == unknown.kernel_id == different.kernel_id == again.kernel_id
    assert not first.cache_hit and not unknown.cache_hit and not different.cache_hit
    assert again.cache_hit
