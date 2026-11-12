"""Production inventory selection and frozen AOT identities without a compiler."""

from tests.production_inventory import PRODUCTION

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer.aot import build
from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID, _contracts
from flashinfer.jit import AscendCBuildConfig
from flashinfer.runtime import SchemaError


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("selector", ["matmul_bias_silu", "flash_attention", "all"])
def test_config_selects_a_recipe_or_all_and_keeps_legacy_default(tmp_path, selector):
    path = tmp_path / "aot.json"
    path.write_text(json.dumps(dict(recipe_id=selector, output_kind="op_package")))
    assert AscendCBuildConfig.from_file(path).recipe_id == selector
    assert AscendCBuildConfig.from_file(ROOT / "examples/ascendc_aot_native.json").recipe_id == "matmul_probe"


@pytest.mark.parametrize("recipe_id,implementation,kernel_id", [
    ("matmul_bias_silu", "native", MATMUL_BIAS_SILU_KERNEL_ID),
    ("flash_attention", "catlass", FLASH_ATTENTION_KERNEL_ID),
])
def test_config_routes_production_recipe_to_exact_frozen_builder(
        monkeypatch, tmp_path, recipe_id, implementation, kernel_id):
    calls = []
    monkeypatch.setattr(build, "build_production_packages", lambda **kwargs: calls.append(kwargs) or (tmp_path,))
    config = AscendCBuildConfig(implementation=implementation, output_kind="op_package", recipe_id=recipe_id)
    assert build.compile_and_package(config, output_dir=tmp_path) == (tmp_path,)
    assert calls[0]["kernel_ids"] == (kernel_id,)
    assert calls[0]["targets"] == ("Ascend950",)


def test_all_config_preserves_target_and_uses_registry_implementations(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(build, "build_production_packages", lambda **kwargs: calls.append(kwargs) or ())
    config = AscendCBuildConfig(output_kind="op_package", recipe_id="all", target_soc="ascend910b1")
    build.compile_and_package(config, output_dir=tmp_path)
    assert calls[0]["targets"] == ("ascend910b1",)
    assert "kernel_ids" not in calls[0]


def test_all_inventory_automatically_includes_new_registered_contract(monkeypatch, tmp_path, capsys):
    import flashinfer.ascendc as ascendc

    contracts = dict(_contracts())
    descriptor, recipe, spec = contracts[MATMUL_BIAS_SILU_KERNEL_ID]
    # This planning-only test models one release toolchain. The real catalog
    # may retain independently frozen releases from different build machines.
    contracts = {key: (d, r, replace(s, environment_fingerprint=spec.environment_fingerprint))
                 for key, (d, r, s) in contracts.items()}
    new_id = recipe.kernel_id.replace("matmul_bias_silu", "future_op")
    future = (replace(descriptor, kernel_id=new_id),
              replace(recipe, kernel_id=new_id, recipe_id="future_op"),
              replace(spec, kernel_id=new_id))
    contracts[new_id] = future
    monkeypatch.setattr(ascendc, "_contracts", lambda: contracts)
    catlass = tmp_path / "catlass"
    (catlass / "include").mkdir(parents=True)
    monkeypatch.setenv("FLASHINFER_CATLASS_ROOT", str(catlass))
    monkeypatch.setattr(build.AscendCToolchain, "discover",
                        lambda *args: SimpleNamespace(fingerprint=spec.environment_fingerprint))
    monkeypatch.setattr(build, "resolve_build_recipe", lambda recipe, **kwargs: SimpleNamespace(
        fingerprint=dict(contracts[recipe.kernel_id][2].input_fingerprints)["resolved_recipe"]))
    backends = []
    monkeypatch.setattr(build, "AscendCBackend", lambda **kwargs: backends.append(kwargs) or object())
    published = []

    def publish(recipe, spec, backend, store, output_dir):
        published.append((recipe, spec))
        return Path(output_dir) / spec.fingerprint

    monkeypatch.setattr(build, "_publish", publish)
    outputs = build.build_production_packages(output_dir=tmp_path / "release")
    assert [recipe.kernel_id for recipe, _ in published] == list(contracts)
    assert all(spec is contracts[recipe.kernel_id][2] for recipe, spec in published)
    assert len(outputs) == len(backends) == len(PRODUCTION) + 1
    assert json.loads(capsys.readouterr().out)["summary"] == dict(requested=len(PRODUCTION) + 1, built=len(PRODUCTION) + 1, failed=0, skipped=0)


def test_all_missing_dependency_reports_unbuilt_kernel_before_compilation(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASHINFER_CATLASS_ROOT", raising=False)
    monkeypatch.setattr(build.AscendCToolchain, "discover", lambda *args: object())
    monkeypatch.setattr(build, "_publish", lambda *args: pytest.fail("must not compile a partial all release"))
    with pytest.raises(SchemaError, match="not built due to missing dependency catlass") as caught:
        build.build_production_packages(output_dir=tmp_path / "release", source_root=tmp_path)
    assert FLASH_ATTENTION_KERNEL_ID in str(caught.value)
    assert "FLASHINFER_CATLASS_ROOT" in str(caught.value)
    assert not (tmp_path / "release").exists()


def test_all_rejects_unsupported_target_instead_of_silently_building_950(monkeypatch, tmp_path):
    monkeypatch.setattr(build.AscendCToolchain, "discover", lambda *args: object())
    with pytest.raises(SchemaError, match="requires target Ascend950"):
        build.build_production_packages(output_dir=tmp_path, targets=("ascend910b1",))
