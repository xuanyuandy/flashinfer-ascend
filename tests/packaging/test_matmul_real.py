"""Real AOT delivery, lazy JIT, numerical and queue interoperability checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from flashinfer.operators.pytest_support import corrupt_package_copy
from tests.support.aot_release import runtime_only_cann

pytestmark = pytest.mark.requires_soc("Ascend950")

from flashinfer._matmul import AOT_KERNEL_ID


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def matmul_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages

    assert torch.npu.is_available(), "requires a real NPU; missing runtime dependencies must fail"
    root = tmp_path_factory.mktemp("matmul-aot")
    release = os.environ.get("FLASHINFER_MATMUL_TEST_RELEASE")
    if release is None:
        release = root / "release"
        build_production_packages(kernel_ids=(AOT_KERNEL_ID,), output_dir=release,
                                  cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", root / "aot-cache"))
    adapter = root / "adapter"
    build_adapter(adapter)
    return dict(FLASHINFER_AOT_PATHS=str(Path(release).resolve()),
                FLASHINFER_ACLNN_ADAPTER_PATH=str(adapter))


def execute(mode, tmp_path, *, release=None, cache=None, **overrides):
    environment = dict(os.environ, FLASHINFER_NPU_CACHE_DIR=str(cache or tmp_path / "jit-cache"),
                       FLASHINFER_NPU_TUNING_DB="", FLASHINFER_NPU_JIT_DISABLE="0",
                       FLASHINFER_NPU_BACKEND="auto")
    if release:
        environment.update(release)
    environment.update(overrides)
    command = [sys.executable, "-B", "-m", "tests.support.matmul_acceptance", mode]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=240)
    (tmp_path / (mode + ".log")).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"') and '"result": "PASS"' in line)
    print(json.dumps(evidence, sort_keys=True))
    return evidence


@pytest.fixture(scope="module")
def warm_matmul_environment(matmul_release, tmp_path_factory):
    root = tmp_path_factory.mktemp("matmul-warm-jit")
    cache = root / "jit-cache"
    execute("cold_jit", root, cache=cache)
    return dict(matmul_release, FLASHINFER_NPU_CACHE_DIR=str(cache),
                FLASHINFER_MATMUL_EXPECT_TRITON_COMPILES="0")


@pytest.mark.real_compile
def test_three_backends_seven_shapes_functional_out_and_same_buffers(warm_matmul_environment, tmp_path):
    evidence = execute("matrix", tmp_path, release=warm_matmul_environment)
    assert len(evidence["errors"]) == 42
    assert evidence["compiler_calls"] == dict(ascendc=0, triton=0, toolchain=0)
    assert evidence["native_source"] == "aot"


@pytest.mark.real_compile
def test_triton_cold_compile_and_compiler_free_process_disk_hit(tmp_path):
    cache = tmp_path / "cold-cache"
    execute("cold_jit", tmp_path, cache=cache)
    execute("cached_jit", tmp_path, cache=cache)
    logs = list((cache / "jit/.objects").glob("*/build.log"))
    assert len(logs) == 1
    assert "compiled triton/gemm/matmul_v1/Ascend950" in logs[0].read_text()


@pytest.mark.real_compile
def test_aot_and_cann_without_visible_compiler(matmul_release, tmp_path):
    evidence = execute("aot_without_compiler", tmp_path, release=matmul_release,
                       FLASHINFER_NPU_JIT_DISABLE="1", **runtime_only_cann(tmp_path))
    assert evidence["compiler_calls"] == dict(ascendc=0, triton=0, toolchain=0)


@pytest.mark.real_compile
@pytest.mark.parametrize("jit_disabled", ["0", "1"])
def test_missing_aot_never_compiles_even_with_global_jit_enabled(tmp_path, jit_disabled):
    empty = tmp_path / "empty-aot"
    empty.mkdir()
    evidence = execute("missing_" + jit_disabled, tmp_path,
                       FLASHINFER_MATMUL_ERROR_ROOT=str(empty), FLASHINFER_NPU_JIT_DISABLE=jit_disabled)
    assert evidence["category"] == "missing" and evidence["fallback"] is False


@pytest.mark.real_compile
def test_corrupt_aot_rejected_without_jit(matmul_release, tmp_path):
    release_root = Path(matmul_release["FLASHINFER_AOT_PATHS"])
    from flashinfer.ascendc import load_contract
    from flashinfer.jit.discovery import AotPackageIndex

    descriptor, _, spec = load_contract(AOT_KERNEL_ID)
    package = AotPackageIndex(roots=(release_root,)).find(
        spec, descriptor=descriptor, target_soc=descriptor.target_soc).directory
    root = corrupt_package_copy(package, tmp_path / "damaged")
    evidence = execute("corrupt", tmp_path, FLASHINFER_MATMUL_ERROR_ROOT=str(root))
    assert evidence["category"] == "corrupt" and evidence["fallback"] is False


@pytest.mark.real_compile
@pytest.mark.parametrize("queue_mode", [0, 1, 2])
def test_mixed_cann_aot_triton_on_nondefault_stream(warm_matmul_environment, tmp_path, queue_mode):
    execute("queue" + str(queue_mode), tmp_path, release=warm_matmul_environment)
