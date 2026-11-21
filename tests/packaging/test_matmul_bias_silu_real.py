"""Production T2' AOT/JIT and numeric acceptance on real Ascend hardware."""

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from flashinfer.operators.pytest_support import corrupt_package_copy
from tests.support.aot_release import installed_release, run_release_scenario, runtime_only_cann

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


def execute(scenario, tmp_path, *, search_root=None):
    env = runtime_only_cann(tmp_path) if scenario == "hit_disabled" else {}
    return run_release_scenario("tests.support.matmul_bias_silu_acceptance", scenario, tmp_path / "cold-store",
                                search_root=search_root, env=env)


@pytest.mark.real_compile
def test_production_aot_without_compiler(production_release, tmp_path):
    with installed_release(production_release):
        evidence = execute("hit_disabled", tmp_path)
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0
    assert not evidence["compiler_path_visible"] and evidence["toolchain_calls"] == 0
    assert not evidence["compiler_home_visible"]


@pytest.mark.real_compile
def test_production_missing_disabled(production_release, tmp_path):
    evidence = execute("missing_disabled", tmp_path)
    assert evidence["category"] == "missing" and not evidence["fallback"]


@pytest.mark.real_compile
def test_production_corrupt_rejects_jit(production_release, tmp_path):
    root = corrupt_package_copy(production_release, tmp_path / "damaged")
    evidence = execute("corrupt", tmp_path, search_root=root)
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
def test_production_shape_rejected_in_dispatch(production_release, tmp_path):
    evidence = execute("out_of_range", tmp_path)
    assert evidence["category"] == "dispatch" and not evidence["runner_resolved"]
