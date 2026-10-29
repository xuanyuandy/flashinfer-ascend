"""Bounded real FA validation; no skips, broad regression or synthetic timings."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def fa_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available() and torch.npu.device_count() == 8
    release = os.environ.get("FLASHINFER_FA_TEST_AOT_PATH")
    adapter = os.environ.get("FLASHINFER_FA_TEST_ADAPTER_PATH")
    if bool(release) != bool(adapter):
        raise RuntimeError("provide both FA test AOT and native adapter paths")
    if not release:
        from flashinfer.aclnn.build import build_adapter
        from flashinfer.aot.build import build_production_packages
        root = tmp_path_factory.mktemp("fa-aot")
        build_adapter(root / "adapter")
        build_production_packages(output_dir=root / "release",
                                  cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", root / "cache"))
        release, adapter = str(root / "release"), str(root / "adapter")
    return dict(FLASHINFER_AOT_PATHS=release, FLASHINFER_ACLNN_ADAPTER_PATH=adapter)


def accept(mode, environment, cache_only=False):
    command = [sys.executable, "-B", "-m", "tests.triton_attention_acceptance", mode]
    if cache_only:
        command.append("cached")
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=180)
    print(result.stdout, end="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"result": "PASS"' in result.stdout


@pytest.mark.real_compile
def test_cold_jit_and_compiler_free_disk_hit(tmp_path):
    environment = dict(os.environ, FLASHINFER_NPU_CACHE_DIR=str(tmp_path / "cold"),
                       FLASHINFER_NPU_TUNING_DB="", FLASHINFER_NPU_JIT_DISABLE="0")
    accept("cold", environment)
    # New process and runner, disk bytes + prebuilt runtime extensions only.
    accept("cold", environment, cache_only=True)
    logs = list((tmp_path / "cold" / "jit" / ".objects").glob("*/build.log"))
    assert len(logs) == 2
    assert all("compiled triton/attention/dense_prefill_v1/Ascend950" in log.read_text() for log in logs)


@pytest.fixture(scope="module")
def warm_environment(fa_release, tmp_path_factory):
    cache = os.environ.get("FLASHINFER_FA_TEST_JIT_CACHE") or tmp_path_factory.mktemp("triton-fa-cache")
    return dict(os.environ, **fa_release, FLASHINFER_NPU_CACHE_DIR=str(cache),
                FLASHINFER_NPU_TUNING_DB="", FLASHINFER_NPU_JIT_DISABLE="0")


@pytest.mark.real_compile
def test_dual_dtype_six_shapes_public_torch_and_out(warm_environment):
    accept("matrix", warm_environment)


@pytest.mark.real_compile
@pytest.mark.parametrize("queue_mode", [0, 1, 2])
def test_mixed_aclnn_triton_torchnpu_queue(warm_environment, queue_mode):
    accept("queue" + str(queue_mode), warm_environment)


@pytest.mark.real_compile
def test_public_api_uses_two_exact_shape_records_with_real_runners(warm_environment, tmp_path):
    # The clock is explicitly synthetic. Numerical execution is real; this test
    # proves routing and does not claim a measured performance champion.
    environment = dict(warm_environment, FLASHINFER_NPU_TUNING_DB=str(tmp_path / "routing.sqlite3"))
    accept("routing", environment)
