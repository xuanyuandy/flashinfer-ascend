"""Production T2' AOT/JIT and numeric acceptance on real Ascend hardware."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from flashinfer._resources import data_root
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_soc("Ascend950")

from tests.support.matmul_bias_silu_acceptance import ROOT


@pytest.fixture(scope="module")
def production_release(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires real NPU; a skip is not T2' acceptance")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    if not (cann / "bin/bisheng").is_file():
        pytest.skip("requires bisheng to build the production AOT fixture")
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    from tests.support.aot_release import build_release

    return build_release(MATMUL_BIAS_SILU_KERNEL_ID, tmp_path_factory.mktemp("matmul-bias-silu"))


@contextmanager
def installed_release(release):
    root = data_root(ROOT) / "op_packages"
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
    command = [sys.executable, "-B", "-m", "tests.support.matmul_bias_silu_acceptance",
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
    """Build a valid package bound to another A5 product identity.

    A generated adapter binds the complete recipe, including its target.
    Rewriting only JSON would produce a corrupt fixture. Compilation here
    prepares the fixture; the runtime assertion still forbids JIT entirely.
    """
    from flashinfer.ascendc import load_contract
    from flashinfer.jit import (AscendCBackend, AscendCToolchain, DiskJitBuildStore,
                               JitSpec, JitSpecRegistry, declare_kernel, resolve_build_recipe)
    from flashinfer.jit.build import _build_record
    from flashinfer.jit.cache import verify_build_directory
    from flashinfer.jit.recipe import BuildRecipe
    from flashinfer.runtime import KernelRegistry

    verify_build_directory(release)
    spec = JitSpec.from_dict(json.loads((release / "manifest.json").read_text())["spec"])
    package = json.loads((release / "op-package.json").read_text())
    recipe = replace(BuildRecipe.from_dict(package["recipe"]), target_soc="Ascend950DT")
    descriptor = replace(load_contract(spec.kernel_id)[0], target_soc=recipe.target_soc)
    toolchain = AscendCToolchain.discover()
    resolved = resolve_build_recipe(recipe, source_root=ROOT,
                                    toolchain_fingerprint=toolchain.fingerprint)
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    spec = declare_kernel(descriptor.kernel_id, backend=backend,
        registry=KernelRegistry((descriptor,)), spec_registry=JitSpecRegistry(),
        output_path="op-package.json", entry_point=recipe.host_entry,
        environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=spec.specialization_fingerprint,
        compile_options=spec.compile_options)
    store = DiskJitBuildStore(root)
    record = _build_record(spec, backend=backend, store=store)
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
