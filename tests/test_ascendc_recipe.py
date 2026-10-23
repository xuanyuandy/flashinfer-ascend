"""T13-I host contracts and failures; real A5 execution is in the example.

These tests need neither the CANN compiler nor initialized CATLASS headers.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from flashinfer.jit import (
    AscendCBackend, AscendCBuildConfig, AscendCToolchain, BuildDependency, BuildRecipe,
    DiskJitBuildStore, JitCompilationError, JitSpecRegistry, declare_kernel,
    load_build_recipes, resolve_build_recipe,
)
from flashinfer.jit.recipe import CATLASS_REVISION
from flashinfer.runtime import (
    Backend, CompileMode, KernelDescriptor, KernelLaunchABI, KernelRegistry, SchemaError,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipes():
    return tuple(r for r in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                 if r.output_kind == "host_shared")


def toolchain():
    # Explicit fixture identity, never a claim that a compiler ran.
    return AscendCToolchain(Path("/missing/cann"), Path("/missing/bisheng"),
                            Path("/missing/cmake"), Path("/missing/c++"), (("fixture", "unit-test"),))


def descriptor(recipe):
    return KernelDescriptor(recipe.kernel_id, "test.t13.matmul_probe", Backend.ASCENDC,
        compile_mode=CompileMode.JIT, target_soc=recipe.target_soc, source=recipe.source,
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry, ("stream", "a", "b", "c")))


def declare(resolved, **kwargs):
    d = descriptor(resolved.recipe)
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain())
    values = dict(backend=backend, registry=KernelRegistry((d,)), spec_registry=JitSpecRegistry(),
                  output_path="probe/kernel.so", entry_point=resolved.recipe.host_entry,
                  environment_fingerprint=toolchain().fingerprint, specialization_fingerprint="a" * 64)
    values.update(kwargs)
    return backend, declare_kernel(d.kernel_id, **values)


def test_manifest_source_closure_and_configuration_switch_are_explicit(recipes):
    native, catlass = recipes
    for recipe in recipes:
        assert BuildRecipe.from_dict(json.loads(json.dumps(recipe.to_dict()))) == recipe
        assert recipe.source.language == "ascendc"
        assert recipe.output_kind == "host_shared"
        for ref in recipe.sources + recipe.headers:
            assert hashlib.sha256((ROOT / ref.locator).read_bytes()).hexdigest() == ref.digest
    assert native.dependencies == ()
    assert catlass.dependencies == (BuildDependency("catlass", CATLASS_REVISION),)
    for variant in ("native", "catlass"):
        config = AscendCBuildConfig.from_file(ROOT / ("examples/ascendc_" + variant + ".json"))
        assert config.select(recipes, "matmul_probe").implementation == variant
    assert AscendCBuildConfig(overrides=(("matmul_probe", "catlass"),)).select(recipes, "matmul_probe") == catlass
    with pytest.raises(SchemaError, match="found 0"):
        AscendCBuildConfig(implementation="native").select((catlass,), "matmul_probe")
    with pytest.raises(SchemaError, match="found 2"):
        AscendCBuildConfig().select((native, native), "matmul_probe")


def test_native_resolution_never_reads_catlass_or_runs_git(recipes, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("native must not inspect an optional dependency")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    # Even resolving the supplied optional root would be a dependency leak.
    class ForbiddenRoot:
        def __fspath__(self):
            forbidden()

    resolved = resolve_build_recipe(recipes[0], source_root=ROOT,
        toolchain_fingerprint=toolchain().fingerprint, dependency_roots={"catlass": ForbiddenRoot()})
    assert resolved.dependencies == ()
    backend, spec = declare(resolved)
    assert dict(spec.input_fingerprints)["build_recipe"] == recipes[0].fingerprint
    assert dict(spec.input_fingerprints)["resolved_recipe"] == resolved.fingerprint
    # Declaring a bound recipe, constructing a store and status identities do no I/O.
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    assert declare(resolved)[1] == spec
    assert backend.backend_id == "ascendc"
    DiskJitBuildStore(tmp_path / "unused")


def test_missing_catlass_fails_only_when_selected(recipes, tmp_path):
    with pytest.raises(SchemaError, match="initialize the pinned submodule explicitly"):
        resolve_build_recipe(recipes[1], source_root=ROOT,
            toolchain_fingerprint=toolchain().fingerprint, dependency_roots={"catlass": tmp_path / "absent"})
    resolve_build_recipe(recipes[0], source_root=ROOT, toolchain_fingerprint=toolchain().fingerprint,
                         dependency_roots={"catlass": tmp_path / "absent"})


def test_actual_dependency_content_changes_only_dependent_cache_identity(recipes, tmp_path, monkeypatch):
    include = tmp_path / "include/catlass"
    include.mkdir(parents=True)
    header = include / "catlass.hpp"
    header.write_text("// first header content\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: CATLASS_REVISION + "\n")

    def resolve(recipe):
        return resolve_build_recipe(recipe, source_root=ROOT,
            toolchain_fingerprint=toolchain().fingerprint, dependency_roots={"catlass": tmp_path})

    native = declare(resolve(recipes[0]))[1]
    first = resolve(recipes[1])
    before = declare(first)[1]
    header.write_text("// dirty header, same declared commit\n")
    second = resolve(recipes[1])
    assert first.recipe.fingerprint == second.recipe.fingerprint
    assert first.fingerprint != second.fingerprint
    assert before.fingerprint != declare(second)[1].fingerprint
    assert native == declare(resolve(recipes[0]))[1]
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "0" * 40)
    with pytest.raises(SchemaError, match="revision mismatch"):
        resolve(recipes[1])


@pytest.mark.parametrize("change", [
    dict(output_kind="op_package"), dict(implementation="catlass"),
    dict(tiling_mode="gm_buffer"), dict(workspace_bytes=8), dict(task_type="mixed"),
    dict(device_entry="bad entry"), dict(definitions=("FI_HOST_ENTRY=other",)),
])
def test_unimplemented_or_incoherent_recipe_protocols_fail_closed(recipes, change):
    with pytest.raises(SchemaError):
        replace(recipes[0], **change)


def test_recipe_and_descriptor_cannot_diverge(recipes):
    recipe = recipes[0]
    d = descriptor(recipe)
    for changed in (replace(d, target_soc="Ascend910B"),
                    replace(d, source=replace(d.source, digest="b" * 64)),
                    replace(d, launch_abi=replace(d.launch_abi, abi_name="other")),
                    replace(d, tiling_schema_version=2)):
        with pytest.raises(SchemaError, match="does not bind"):
            recipe.validate_descriptor(changed)


def test_all_recipe_flags_and_protocols_are_in_spec_identity(recipes):
    recipe = recipes[0]

    def unit(r):
        return declare(resolve_build_recipe(r, source_root=ROOT, toolchain_fingerprint=toolchain().fingerprint))[1]

    before = unit(recipe)
    for changes in (dict(compile_options=("-O0",)), dict(link_options=("-Wl,--as-needed",)),
                    dict(definitions=("PROBE_VARIANT=1",)), dict(version=2), dict(block_dim=2)):
        assert unit(replace(recipe, **changes)).fingerprint != before.fingerprint
    resolved = resolve_build_recipe(recipe, source_root=ROOT, toolchain_fingerprint=toolchain().fingerprint)
    for changes in (dict(environment_fingerprint="b" * 64), dict(entry_point="other"),
                    dict(output_path="kernel.o"), dict(compile_options=("-I/untracked",))):
        with pytest.raises(SchemaError):
            declare(resolved, **changes)


def test_modified_source_is_rejected_before_any_compiler_runs(recipes, tmp_path, monkeypatch):
    resolved = resolve_build_recipe(recipes[0], source_root=ROOT, toolchain_fingerprint=toolchain().fingerprint)
    backend, spec = declare(resolved)
    # Move the frozen contract's source root to a corrupt copy. Its fingerprint
    # intentionally depends on contents, not absolute location.
    primary = tmp_path / "inputs" / recipes[0].source.locator
    primary.parent.mkdir(parents=True)
    primary.write_text("// different bytes\n")
    backend = AscendCBackend(recipes=(replace(resolved, source_root=tmp_path / "inputs"),), toolchain=toolchain())
    monkeypatch.setattr(AscendCToolchain, "validate", lambda self: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not invoke compiler"))
    with pytest.raises(JitCompilationError, match="SourceRef digest mismatch") as caught:
        backend.compile(spec, build_directory=tmp_path / "staging")
    assert caught.value.log_path.is_file()
    assert not (tmp_path / "staging/ascendc-build").exists()
    assert not (tmp_path / "staging/probe/kernel.so").exists()


def test_no_cann_discovery_has_an_explicit_failure(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    with pytest.raises(SchemaError, match="requires CANN"):
        AscendCToolchain.discover()


def test_public_recipe_interfaces_are_in_parity_ledger():
    from flashinfer.parity import load_packaged_manifest

    entries = {e.local: e for e in load_packaged_manifest("all").entries}
    for symbol in ("recipe.BuildRecipe", "recipe.BuildDependency", "recipe.ResolvedBuildRecipe",
                   "recipe.AscendCBuildConfig", "recipe.load_build_recipes", "recipe.resolve_build_recipe",
                   "ascendc.AscendCToolchain"):
        assert entries["flashinfer.jit." + symbol].api_kind == "npu_extension"
