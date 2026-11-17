"""A shared A2/A3 wheel keeps distinct per-SoC build identities."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer import _wheel


@pytest.fixture(autouse=True)
def clean_selection(monkeypatch):
    for key in ("FLASHINFER_AOT_SOCS", "FLASHINFER_AOT_PREBUILT", "FLASHINFER_AOT_CONFIG"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("soc,family,name", [
    ("Ascend910B", "a2a3", "flashinfer_ascend910b"),
    ("Ascend910C", "a2a3", "flashinfer_ascend910b"),
    ("Ascend950", "ascend950", "flashinfer_ascend950"),
])
def test_local_and_explicit_selection_keep_the_family_in_the_distribution(monkeypatch, soc, family, name):
    monkeypatch.setattr(_wheel, "current_hardware", lambda: SimpleNamespace(soc_model=soc))
    assert _wheel.wheel_target() == family
    assert _wheel.distribution_name(soc) == name
    monkeypatch.setenv("FLASHINFER_AOT_SOCS", soc)
    monkeypatch.setattr(_wheel, "current_hardware", lambda: pytest.fail("explicit release target probed NPU"))
    assert _wheel.wheel_target() == family


def test_unknown_local_hardware_needs_an_explicit_release_target(monkeypatch):
    monkeypatch.setattr(_wheel, "current_hardware", lambda: SimpleNamespace(soc_model="unknown"))
    with pytest.raises(RuntimeError, match="cannot detect"):
        _wheel.wheel_target()


@pytest.mark.parametrize("soc", ["Ascend910B,Ascend950", "Ascend910B,Ascend910B", "Ascend310P"])
def test_a_single_wheel_rejects_mixed_families_duplicates_and_unknown_chips(monkeypatch, soc):
    monkeypatch.setenv("FLASHINFER_AOT_SOCS", soc)
    with pytest.raises(ValueError):
        _wheel.wheel_target()


def test_a2_a3_combined_selection_is_one_shared_family(monkeypatch):
    monkeypatch.setenv("FLASHINFER_AOT_SOCS", "Ascend910B,Ascend910C")
    assert _wheel.wheel_target() == "a2a3"
    assert _wheel.family_socs("a2a3") == ("Ascend910B", "Ascend910C")


def test_prebuilt_target_is_derived_without_probing_a_device(monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"spec": {"target_soc": "Ascend950"}}))
    monkeypatch.setenv("FLASHINFER_AOT_PREBUILT", str(tmp_path))
    monkeypatch.setattr(_wheel, "current_hardware", lambda: pytest.fail("PREBUILT probed NPU"))
    assert _wheel.wheel_target() == "ascend950"


def test_pruning_retains_both_soc_catalogs_without_faking_binary_compatibility(tmp_path):
    data = tmp_path / "flashinfer/data"
    (data / "kernels").mkdir(parents=True)
    recipes = data / "build/csrc/recipes"
    recipes.mkdir(parents=True)
    entries = [dict(kernel_id="a2/one", target_soc="Ascend910B"),
               dict(kernel_id="a2/two", target_soc="Ascend910B"),
               dict(kernel_id="a3/one", target_soc="Ascend910C"),
               dict(kernel_id="a5/one", target_soc="Ascend950")]
    (data / "kernels/registry.json").write_text(json.dumps(dict(schema_version=3, kernels=entries)))
    (data / "kernels/ascendc_specs.json").write_text(json.dumps(dict(schema_version=1,
        kernels=[dict(kernel_id=r["kernel_id"], recipe=r) for r in entries])))
    (data / "implementations.json").write_text(json.dumps(dict(schema_version=1,
        implementations=[dict(kernel_id=r["kernel_id"], backend="ascendc") for r in entries]
        + [dict(kernel_id="cann/one", backend="cann")])))
    for r in entries:
        (recipes / (r["kernel_id"].replace("/", "_") + ".json")).write_text(json.dumps([r]))
    selected = ("a2/one", "a2/two", "a3/one")
    _wheel.finalize_resources(tmp_path / "flashinfer", target="a2a3", kernel_ids=selected)
    for name in ("registry.json", "ascendc_specs.json"):
        actual = json.loads((data / "kernels" / name).read_text())["kernels"]
        assert {r["kernel_id"] for r in actual} == set(selected)
    actual = json.loads((data / "implementations.json").read_text())["implementations"]
    assert {r["kernel_id"] for r in actual} == {*selected, "cann/one"}
    assert {p.name for p in recipes.iterdir()} == {"a2_one.json", "a2_two.json", "a3_one.json"}
    assert json.loads((data / "wheel.json").read_text()) == dict(schema_version=2,
        family="a2a3", soc_models=["Ascend910B", "Ascend910C"],
        distribution="flashinfer_ascend910b", kernel_ids=sorted(selected))


def test_release_cli_deduplicates_a2_a3_into_one_wheel(monkeypatch, tmp_path):
    from flashinfer.aot import wheels
    calls = []
    monkeypatch.setenv("FLASHINFER_AOT_ALL", "0")
    monkeypatch.setattr(wheels.subprocess, "run", lambda argv, **kwargs:
        calls.append((argv, kwargs)) or SimpleNamespace(returncode=0))
    assert wheels.main(["--soc", "Ascend910B", "--soc", "Ascend910C", "--soc", "Ascend950", "--output-dir", str(tmp_path)]) == 0
    assert [options["env"]["FLASHINFER_AOT_SOCS"] for _, options in calls] == ["a2a3", "ascend950"]
    assert [options["env"]["FLASHINFER_AOT_ALL"] for _, options in calls] == ["1", "1"]
    assert calls[0][0] != calls[1][0]


@pytest.fixture
def prebuilt_binding(tmp_path, monkeypatch):
    from flashinfer.ascendc.gemm.matmul.implementation import IMPLEMENTATION
    from flashinfer.jit import core, cache, op_package

    root = Path(__file__).resolve().parents[2]
    prepared = IMPLEMENTATION.prepare_outputs(root)
    descriptor, recipe = prepared["descriptor"], prepared["recipe"]
    package = tmp_path / "flashinfer"
    data = package / "data"
    (data / "kernels").mkdir(parents=True)
    recipes = data / "build/csrc/recipes"
    recipes.mkdir(parents=True)
    (recipes / "matmul.json").write_text(json.dumps([recipe.to_dict()]))
    (data / "kernels/registry.json").write_text(json.dumps(dict(schema_version=3,
        generated_at="fixture", kernels=[descriptor.to_dict()])))
    spec = core.JitSpec(name="wheel.prebuilt", domain="kernel", generator_id="ascendc",
        generator_version="2", target_soc=recipe.target_soc,
        environment_fingerprint="e" * 64, specialization_fingerprint="f" * 64,
        entry_points=(recipe.host_entry,), kernel_id=recipe.kernel_id,
        source=recipe.source, output_path="op-package.json",
        input_fingerprints=(("kernel", descriptor.fingerprint), ("build_recipe", recipe.fingerprint)))
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps(dict(spec=spec.to_dict())))
    # The immutable ELF/receipt verifier is independently covered; this fixture
    # exercises binding a verified release to a fresh authoring catalog.
    monkeypatch.setattr(cache, "verify_build_directory", lambda _: None)
    monkeypatch.setattr(op_package, "verify_op_package", lambda *args: ({}, recipe))
    return package, release, recipe, spec


def test_prebuilt_supplies_frozen_metadata_without_local_compilation(prebuilt_binding):
    package, release, recipe, spec = prebuilt_binding
    path = package / "data/kernels/ascendc_specs.json"
    assert not path.exists()
    _wheel.restore_prebuilt_specs(package, (release,))
    assert json.loads(path.read_text()) == json.loads(json.dumps(dict(schema_version=1,
        kernels=[dict(kernel_id=recipe.kernel_id, recipe=recipe.to_dict(), spec=spec.to_dict())])))


def test_prebuilt_rejects_changed_capabilities_before_publishing_specs(prebuilt_binding):
    package, release, _, _ = prebuilt_binding
    path = package / "data/kernels/registry.json"
    payload = json.loads(path.read_text())
    payload["kernels"][0]["priority"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(_wheel.SchemaError, match="current operator declaration"):
        _wheel.restore_prebuilt_specs(package, (release,))
    assert not (package / "data/kernels/ascendc_specs.json").exists()


def test_shared_prebuilt_cannot_omit_an_authored_a3_counterpart(tmp_path):
    data = tmp_path / 'flashinfer/data'
    recipes = data / 'build/csrc/recipes'
    recipes.mkdir(parents=True)
    entries = [dict(kernel_id='a2/one', recipe_id='one', implementation='native', target_soc='Ascend910B'),
               dict(kernel_id='a3/one', recipe_id='one', implementation='native', target_soc='Ascend910C')]
    (recipes / 'one.json').write_text(json.dumps(entries))
    before = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='missing authored recipe variants: a3/one'):
        _wheel.finalize_resources(tmp_path / 'flashinfer', target='a2a3', kernel_ids=('a2/one',))
    assert {p: p.read_bytes() for p in data.rglob('*') if p.is_file()} == before
