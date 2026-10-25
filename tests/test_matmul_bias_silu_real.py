"""Production T2' AOT/JIT and numeric acceptance on real Ascend hardware."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.matmul_bias_silu_acceptance import ROOT


@pytest.fixture(scope="module")
def production_release(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires real NPU; a skip is not T2' acceptance")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    if not (cann / "bin/bisheng").is_file():
        pytest.skip("requires bisheng to build the production AOT fixture")
    from tools.build_matmul_bias_silu import build_release

    return build_release(tmp_path_factory.mktemp("matmul-bias-silu"))


@contextmanager
def installed_release(release):
    root = ROOT / "flashinfer/data/op_packages"
    created = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / release.name
    assert not destination.exists(), "never overwrite an existing release"
    shutil.copytree(release, destination)
    try:
        yield destination
    finally:
        shutil.rmtree(destination)
        if created:
            root.rmdir()


def execute(scenario, tmp_path, *, search_root=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("FLASHINFER_AOT_PATHS", None)
    environment.pop("FLASHINFER_NPU_JIT_DISABLE", None)
    if search_root is not None:
        environment["FLASHINFER_AOT_PATHS"] = str(search_root)
    if scenario == "hit_disabled":
        # A runtime-only CANN view: no compiler executable, build tools, or
        # compiler headers. Keep the platform data required by host tiling.
        original_home = Path(environment["ASCEND_HOME_PATH"])
        runtime_home = tmp_path / "runtime-only-cann"
        runtime_home.mkdir()
        for name in ("lib64", "share", "conf", "opp", "runtime"):
            (runtime_home / name).symlink_to(original_home / name, target_is_directory=True)
        # torch_npu checks this directory's presence at import, not its contents.
        (runtime_home / "compiler").mkdir()
        assert not (runtime_home / "bin/bisheng").exists()
        environment["ASCEND_HOME_PATH"] = str(runtime_home)
        environment["PATH"] = "/usr/bin:/bin"
    command = [sys.executable, "-B", "-m", "tests.matmul_bias_silu_acceptance",
               scenario, str(tmp_path / "cold-store")]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"') and '"scenario"' in line)
    assert evidence["result"] == "PASS" and evidence["scenario"] == scenario
    print("COMMAND " + " ".join(command))
    print(json.dumps(evidence, sort_keys=True))
    return evidence


@pytest.mark.real_compile
def test_production_aot_without_compiler(production_release, tmp_path):
    with installed_release(production_release):
        evidence = execute("hit_disabled", tmp_path)
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0
    assert not evidence["compiler_path_visible"] and evidence["toolchain_calls"] == 0
    assert not evidence["compiler_home_visible"]


@pytest.mark.real_compile
def test_production_missing_real_jit(production_release, tmp_path):
    evidence = execute("missing_enabled", tmp_path)
    assert evidence["source"] == "jit" and evidence["compiler_calls"] == 1


@pytest.mark.real_compile
def test_production_missing_disabled(production_release, tmp_path):
    evidence = execute("missing_disabled", tmp_path)
    assert evidence["category"] == "missing" and not evidence["fallback"]


@pytest.mark.real_compile
def test_production_corrupt_rejects_jit(production_release, tmp_path):
    damaged = tmp_path / "damaged" / production_release.name
    shutil.copytree(production_release, damaged)
    package = json.loads((damaged / "op-package.json").read_text())
    library = damaged / package["api_library"]
    payload = library.read_bytes()
    library.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    evidence = execute("corrupt", tmp_path, search_root=damaged.parent)
    assert evidence["policy"] == "enabled" and evidence["compiler_calls"] == 0


def incompatible_release(release, root):
    """Re-publish valid inventory with another A5 product identity.

    The numerical executable is not run on the mismatched product. This
    deliberate metadata fixture isolates the target check from corruption.
    """
    from flashinfer.jit import DiskJitBuildStore, JitSpec
    from flashinfer.jit.build import _verify_output
    from flashinfer.jit.cache import verify_build_directory
    from flashinfer.jit.recipe import BuildRecipe

    verify_build_directory(release)
    spec = JitSpec.from_dict(json.loads((release / "manifest.json").read_text())["spec"])
    package = json.loads((release / "op-package.json").read_text())
    recipe = replace(BuildRecipe.from_dict(package["recipe"]), target_soc="Ascend950DT")
    inputs = dict(spec.input_fingerprints, build_recipe=recipe.fingerprint)
    spec = replace(spec, target_soc=recipe.target_soc, input_fingerprints=tuple(sorted(inputs.items())))
    package.update(recipe=recipe.to_dict(), spec_fingerprint=spec.fingerprint)
    store = DiskJitBuildStore(root)
    with store.staging(spec) as staging:
        shutil.copytree(release, staging, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("manifest.json", "build.log"))
        (staging / "op-package.json").write_text(json.dumps(package, sort_keys=True))
        artifact = _verify_output(spec, staging)
        record = store.publish(spec, staging, artifact)
    return store.directory(record)


@pytest.mark.real_compile
def test_production_incompatible_soc(production_release, tmp_path):
    other = incompatible_release(production_release, tmp_path / "other-product")
    evidence = execute("incompatible", tmp_path, search_root=other)
    assert "Ascend950DT" in evidence["diagnostic"] and evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_production_numerical_fp32_fp64(production_release, tmp_path):
    evidence = execute("numerical", tmp_path, search_root=production_release)
    assert len(evidence["errors"]) == 8


@pytest.mark.real_compile
def test_production_shape_rejected_in_dispatch(production_release, tmp_path):
    evidence = execute("out_of_range", tmp_path)
    assert evidence["category"] == "dispatch" and not evidence["runner_resolved"]
