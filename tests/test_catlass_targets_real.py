"""Compile both supported CATLASS targets; execute only on matching hardware."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from flashinfer.jit.recipe import CATLASS_REVISION


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Ascend910B", "Ascend910C", "Ascend950")


@pytest.fixture(scope="module")
def catlass_releases(tmp_path_factory):
    from flashinfer.aot.build import build_production_packages
    from flashinfer.aclnn.build import build_adapter

    directory = tmp_path_factory.mktemp("catlass-targets")
    release = directory / "release"
    ids = tuple("ascendc/t4/flash_attention/catlass/op_package/" + t for t in TARGETS)
    packages = build_production_packages(kernel_ids=ids, output_dir=release,
        cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", directory / "cache"))
    adapter = directory / "adapter"
    build_adapter(adapter)
    return release, adapter, dict(zip(TARGETS, packages))


@pytest.mark.real_compile
@pytest.mark.parametrize("target", TARGETS)
def test_catlass_attention_target_and_generated_binding(catlass_releases, target):
    from flashinfer.jit.cache import verify_build_directory
    from flashinfer.jit.core import JitSpec
    from flashinfer.jit.op_package import verify_op_package

    package = catlass_releases[2][target]
    verify_build_directory(package)
    spec = JitSpec.from_dict(json.loads((package / "manifest.json").read_text())["spec"])
    metadata, _ = verify_op_package(package, spec)
    assert spec.target_soc == target
    assert metadata["schema_version"] == 2
    assert metadata["recipe"]["dependencies"][0]["revision"] == CATLASS_REVISION
    binding = json.loads((package / metadata["native_adapter"]).read_text())
    assert binding["kernel_id"] == spec.kernel_id
    assert binding["host_entry"] == "aclnnFiFlashAttention"


@pytest.mark.real_compile
def test_catlass_local_chip_numerics_out_and_current_stream(catlass_releases, tmp_path):
    release, adapter, _ = catlass_releases
    environment = dict(os.environ, FLASHINFER_AOT_PATHS=str(release),
        FLASHINFER_ACLNN_ADAPTER_PATH=str(adapter), FLASHINFER_NPU_JIT_DISABLE="1")
    command = [sys.executable, "-B", "-m", "tests.catlass_target_acceptance"]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=300)
    (tmp_path / "numerics.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    record = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"'))
    assert record["result"] == "PASS" and len(record["errors"]) == 12


@pytest.mark.real_compile
@pytest.mark.parametrize("target", TARGETS)
def test_catlass_public_matmul_probe_builds_for_each_target(tmp_path, target):
    from flashinfer.aot.build import compile_and_package

    packages = compile_and_package(ROOT / "examples/ascendc_aot_catlass.json",
        targets=(target,), output_dir=tmp_path / "release",
        cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", tmp_path / "cache"),
        catlass_root=os.environ.get("FLASHINFER_CATLASS_ROOT", ROOT / "3rdparty/catlass"))
    assert len(packages) == 1
    metadata = json.loads((packages[0] / "op-package.json").read_text())
    assert metadata["recipe"]["target_soc"] == target
    assert metadata["recipe"]["implementation"] == "catlass"
    assert metadata["schema_version"] == 2
