"""The seven T4 production acceptance scenarios on real Ascend950 hardware."""

import os
from pathlib import Path

import pytest

from flashinfer.operators.pytest_support import corrupt_package_copy
from tests.support.aot_release import installed_release, run_release_scenario, runtime_only_cann

pytestmark = pytest.mark.requires_soc("Ascend950")

from tests.support.flash_attention_acceptance import CATLASS_ROOT, PYTHON


@pytest.fixture(scope="module")
def flash_attention_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "requires real NPU; a skip is not T4 acceptance"
    assert torch.npu.device_count() == 8, "expected the declared eight Ascend devices"
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    assert (cann / "bin/bisheng").is_file(), "real bisheng is required to build the AOT fixture"
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    from tests.support.aot_release import build_release

    original = os.environ.get("FLASHINFER_CATLASS_ROOT")
    os.environ["FLASHINFER_CATLASS_ROOT"] = str(CATLASS_ROOT)
    try:
        return build_release(FLASH_ATTENTION_KERNEL_ID, tmp_path_factory.mktemp("flash-attention-release"))
    finally:
        if original is None:
            os.environ.pop("FLASHINFER_CATLASS_ROOT", None)
        else:
            os.environ["FLASHINFER_CATLASS_ROOT"] = original


def execute(scenario, tmp_path, *, search_root=None):
    env = {"FLASHINFER_CATLASS_ROOT": str(CATLASS_ROOT), "FLASHINFER_TEST_CATLASS_ROOT": str(CATLASS_ROOT)}
    if scenario == "hit_disabled":
        env.update(runtime_only_cann(tmp_path))
        absent = tmp_path / "absent-build-catlass"
        assert not absent.exists()
        env["FLASHINFER_CATLASS_ROOT"] = str(absent)
    return run_release_scenario("tests.support.flash_attention_acceptance", scenario, tmp_path / "cold-store",
                                search_root=search_root, env=env, python=PYTHON, timeout=300)


@pytest.mark.real_compile
def test_flash_attention_aot_without_compiler(flash_attention_release, tmp_path):
    with installed_release(flash_attention_release):
        evidence = execute("hit_disabled", tmp_path)
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0
    assert evidence["toolchain_calls"] == 0
    assert not evidence["compiler_path_visible"] and not evidence["compiler_home_visible"]
    assert not evidence["catlass_root_available"]


@pytest.mark.real_compile
def test_flash_attention_missing_disabled(flash_attention_release, tmp_path):
    evidence = execute("missing_disabled", tmp_path)
    assert evidence["category"] == "missing" and not evidence["fallback"]
    assert evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_flash_attention_corrupt_rejects_jit(flash_attention_release, tmp_path):
    root = corrupt_package_copy(flash_attention_release, tmp_path / "damaged")
    evidence = execute("corrupt", tmp_path, search_root=root)
    assert evidence["policy"] == "enabled" and evidence["compiler_calls"] == 0
    assert evidence["category"] == "corrupt" and not evidence["fallback"]


@pytest.mark.real_compile
def test_flash_attention_envelope_rejected_in_dispatch(flash_attention_release, tmp_path):
    evidence = execute("out_of_range", tmp_path)
    assert evidence["category"] == "dispatch" and not evidence["runner_resolved"]
    assert evidence["compiler_calls"] == 0 and len(evidence["rejected"]) == 13


@pytest.mark.real_compile
def test_flash_attention_missing_catlass_dependency(tmp_path):
    evidence = execute("missing_dependency", tmp_path)
    assert evidence["category"] == "dependency" and evidence["compiler_calls"] == 0
    assert evidence["stage"] == "build_source_resolution"
