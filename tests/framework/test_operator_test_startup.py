import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer.operators import pytest_support
from tests.conftest import pytest_sessionstart


@pytest.mark.parametrize("scope, multi_target", [("Ascend950", False), ("auto", False), ("auto", True)])
def test_prebuilt_vendors_are_visible_when_selecting_the_test_device(monkeypatch, tmp_path, scope, multi_target):
    release = str(tmp_path / "release")
    vendor = "flashinfer_quant_flash_attn"
    current = str(tmp_path / "new" / vendor)
    inherited = str(tmp_path / "old" / vendor)
    other = str(tmp_path / "other_vendor")
    monkeypatch.setenv("FLASHINFER_OPERATOR_TEST_RELEASE", release)
    monkeypatch.setenv("FLASHINFER_TEST_NPU_DEVICE", "0")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", os.pathsep.join((inherited, other)))

    def resolve(path, aliases, *, target_soc):
        assert path == release and Path(aliases).is_dir()
        assert target_soc == (scope if scope != "auto" else None)
        vendors = [("Ascend950", current)]
        if multi_target:
            vendors.append(("Ascend910B", str(tmp_path / "a2" / vendor)))
        return vendors

    selections = []

    def select_device(index):
        assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == os.pathsep.join((current, other))
        selections.append(index)

    monkeypatch.setattr(pytest_support, "operator_test_vendors", resolve)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "Ascend950\n")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(
        device_count=lambda: 1, set_device=select_device,
    )))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    cleanups = []
    config = SimpleNamespace(
        getoption=lambda name: scope if name == "--soc" else False,
        add_cleanup=cleanups.append,
    )

    try:
        pytest_sessionstart(SimpleNamespace(config=config))
        assert selections == [0]
    finally:
        for cleanup in reversed(cleanups):
            cleanup()
