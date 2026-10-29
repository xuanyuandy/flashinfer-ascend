"""The seven T4 production acceptance scenarios on real Ascend950 hardware."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.flash_attention_acceptance import CATLASS_ROOT, GOLDEN_SHAPES, PYTHON, ROOT


@pytest.fixture(scope="module")
def flash_attention_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "requires real NPU; a skip is not T4 acceptance"
    assert torch.npu.device_count() == 8, "expected the declared eight Ascend devices"
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    assert (cann / "bin/bisheng").is_file(), "real bisheng is required to build the AOT fixture"
    from tools.build_flash_attention import build_release

    original = os.environ.get("FLASHINFER_CATLASS_ROOT")
    os.environ["FLASHINFER_CATLASS_ROOT"] = str(CATLASS_ROOT)
    try:
        return build_release(tmp_path_factory.mktemp("flash-attention-release"))
    finally:
        if original is None:
            os.environ.pop("FLASHINFER_CATLASS_ROOT", None)
        else:
            os.environ["FLASHINFER_CATLASS_ROOT"] = original


@contextmanager
def installed_release(release):
    destination_root = ROOT / "flashinfer/data/op_packages"
    created = not destination_root.exists()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / release.name
    assert not destination.exists(), "never overwrite an existing release"
    shutil.copytree(release, destination)
    try:
        yield destination
    finally:
        shutil.rmtree(destination)
        if created:
            destination_root.rmdir()


def execute(scenario, tmp_path, *, search_root=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       FLASHINFER_CATLASS_ROOT=str(CATLASS_ROOT),
                       FLASHINFER_TEST_CATLASS_ROOT=str(CATLASS_ROOT))
    environment.pop("FLASHINFER_AOT_PATHS", None)
    environment.pop("FLASHINFER_NPU_JIT_DISABLE", None)
    if search_root is not None:
        environment["FLASHINFER_AOT_PATHS"] = str(search_root)
    if scenario == "hit_disabled":
        original_home = Path(environment["ASCEND_HOME_PATH"])
        runtime_home = tmp_path / "runtime-only-cann"
        runtime_home.mkdir()
        for name in ("lib64", "share", "conf", "opp", "runtime"):
            (runtime_home / name).symlink_to(original_home / name, target_is_directory=True)
        (runtime_home / "compiler").mkdir()
        assert not (runtime_home / "bin/bisheng").exists()
        environment["ASCEND_HOME_PATH"] = str(runtime_home)
        environment["PATH"] = "/usr/bin:/bin"
        absent_catlass = tmp_path / "absent-build-catlass"
        assert not absent_catlass.exists()
        environment["FLASHINFER_CATLASS_ROOT"] = str(absent_catlass)
    command = [str(PYTHON), "-B", "-m", "tests.flash_attention_acceptance",
               scenario, str(tmp_path / "cold-store")]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"') and '"scenario"' in line)
    assert evidence["result"] == "PASS" and evidence["scenario"] == scenario
    print("COMMAND " + " ".join(command))
    print(json.dumps(evidence, sort_keys=True))
    return evidence


@pytest.mark.real_compile
def test_flash_attention_golden_multiple_shapes(flash_attention_release, tmp_path):
    evidence = execute("numerical", tmp_path, search_root=flash_attention_release)
    assert len(evidence["errors"]) == len(GOLDEN_SHAPES) * 2
    assert {entry["dtype"] for entry in evidence["errors"]} == {"float16", "bfloat16"}
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_flash_attention_aot_without_compiler(flash_attention_release, tmp_path):
    with installed_release(flash_attention_release):
        evidence = execute("hit_disabled", tmp_path)
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0
    assert evidence["toolchain_calls"] == 0
    assert not evidence["compiler_path_visible"] and not evidence["compiler_home_visible"]
    assert not evidence["catlass_root_available"]


@pytest.mark.real_compile
def test_flash_attention_missing_real_jit(flash_attention_release, tmp_path):
    evidence = execute("missing_enabled", tmp_path)
    assert evidence["source"] == "jit" and evidence["compiler_calls"] == 1
    assert evidence["toolchain_calls"] == 1


@pytest.mark.real_compile
def test_flash_attention_missing_disabled(flash_attention_release, tmp_path):
    evidence = execute("missing_disabled", tmp_path)
    assert evidence["category"] == "missing" and not evidence["fallback"]
    assert evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_flash_attention_corrupt_rejects_jit(flash_attention_release, tmp_path):
    damaged = tmp_path / "damaged" / flash_attention_release.name
    shutil.copytree(flash_attention_release, damaged)
    package = json.loads((damaged / "op-package.json").read_text())
    library = damaged / package["api_library"]
    payload = library.read_bytes()
    library.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    evidence = execute("corrupt", tmp_path, search_root=damaged.parent)
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
