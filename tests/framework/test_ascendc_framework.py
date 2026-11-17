"""Catalog-driven authoring/builds without adding a production registration."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer.aot import build, declarations
from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, load_contract
from flashinfer.jit import AscendCBackend, BuildDependency
from flashinfer.jit.dependencies import dependency_roots
from flashinfer.runtime import SchemaError
from flashinfer._resources import recipe_root


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def authoring(tmp_path, monkeypatch):
    root = tmp_path / "project"
    shutil.copytree(ROOT / "csrc", root / "csrc")
    shutil.copytree(recipe_root(ROOT), root / "csrc/recipes", dirs_exist_ok=True)
    descriptor, recipe, spec = load_contract(MATMUL_BIAS_SILU_KERNEL_ID)
    for reference in recipe.extra_build_files:
        target = root / reference.locator
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / reference.locator, target)
    data = root / "flashinfer/data/kernels"
    registry = data / "registry.json"
    frozen = data / "ascendc_specs.json"
    record = descriptor.to_dict()
    record["graph_capability"] = descriptor.graph_capability.value
    write_json(
        registry, dict(schema_version=3, generated_at="fixture", kernels=[record])
    )
    write_json(
        frozen,
        dict(
            schema_version=1,
            kernels=[
                dict(
                    kernel_id=descriptor.kernel_id,
                    recipe=recipe.to_dict(),
                    spec=spec.to_dict(),
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        declarations.AscendCToolchain,
        "discover",
        lambda *args: SimpleNamespace(fingerprint=spec.environment_fingerprint),
    )
    monkeypatch.setattr(
        AscendCBackend,
        "compile",
        lambda *a, **k: pytest.fail("freezing must not compile"),
    )
    return SimpleNamespace(
        root=root,
        data=data,
        registry=registry,
        frozen=frozen,
        descriptor=descriptor,
        recipe=recipe,
        spec=spec,
    )


def add_future_operator(project):
    original = project.recipe
    kernel_id = original.kernel_id.replace("matmul_bias_silu", "future_matmul")
    recipe = replace(
        original,
        kernel_id=kernel_id,
        recipe_id="future_matmul",
        package_vendor="flashinfer_future_matmul",
        package_op_type="FiFutureMatmul",
        host_entry="aclnnFiFutureMatmul",
        device_entry="fi_future_matmul",
        launch_abi="aclnn.future_matmul.v2",
    )
    descriptor = replace(
        project.descriptor,
        kernel_id=kernel_id,
        op="example.future_matmul",
        capability=replace(project.descriptor.capability, op="example.future_matmul"),
        launch_abi=replace(
            project.descriptor.launch_abi,
            abi_name=recipe.launch_abi,
            entry_point=recipe.host_entry,
        ),
    )
    payload = json.loads(project.registry.read_text())
    record = descriptor.to_dict()
    record["graph_capability"] = descriptor.graph_capability.value
    payload["kernels"].append(record)
    write_json(project.registry, payload)
    write_json(project.root / "csrc/recipes/future_matmul.json", [recipe.to_dict()])
    return descriptor, recipe


def test_refreeze_is_byte_identical_and_never_compiles(authoring):
    before = (authoring.registry.read_bytes(), authoring.frozen.read_bytes())
    (result,) = declarations.freeze_production_kernels(source_root=authoring.root)
    assert result == (authoring.descriptor, authoring.recipe, authoring.spec)
    assert (authoring.registry.read_bytes(), authoring.frozen.read_bytes()) == before


def test_new_operator_needs_only_recipe_and_central_descriptor(authoring, monkeypatch):
    import flashinfer.ascendc as ascendc

    descriptor, recipe = add_future_operator(authoring)
    prepared = declarations.freeze_production_kernels(source_root=authoring.root)
    assert [d.kernel_id for d, _, _ in prepared] == [
        authoring.recipe.kernel_id,
        recipe.kernel_id,
    ]
    with monkeypatch.context() as scoped:
        from flashinfer.ascendc import _catalog

        scoped.setattr(_catalog, "_DATA", authoring.data)
        scoped.setattr(build, "_source_root", lambda: authoring.root)
        ascendc._contracts.cache_clear()
        try:
            bound = {d.kernel_id: d for d in ascendc.ascendc_descriptors()}
            assert set(bound) == {authoring.recipe.kernel_id, recipe.kernel_id}
            runner = bound[recipe.kernel_id].runner
            assert bound[recipe.kernel_id].fingerprint == descriptor.fingerprint
            assert runner.provenance is None
            backend = runner.jit_backend_factory()
            assert backend.backend_id == "ascendc"
            assert backend._recipes[recipe.kernel_id].recipe == recipe
        finally:
            ascendc._contracts.cache_clear()


def test_source_refresh_changes_spec_without_changing_capability_or_abi(authoring):
    source = authoring.root / authoring.recipe.source.locator
    source.write_bytes(source.read_bytes() + b"\n// changed source identity\n")
    ref = replace(
        authoring.recipe.source, digest=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    recipe = replace(
        authoring.recipe,
        source=ref,
        sources=tuple(
            ref if item == authoring.recipe.source else item
            for item in authoring.recipe.sources
        ),
    )
    write_json(
        authoring.root / "csrc/recipes/matmul_bias_silu.json", [recipe.to_dict()]
    )
    ((descriptor, actual, spec),) = declarations.freeze_production_kernels(
        source_root=authoring.root
    )
    assert descriptor.source == actual.source == ref
    assert descriptor.capability == authoring.descriptor.capability
    assert descriptor.launch_abi == authoring.descriptor.launch_abi
    assert descriptor.graph_capability == authoring.descriptor.graph_capability
    assert spec.fingerprint != authoring.spec.fingerprint
    assert spec.specialization_fingerprint == authoring.spec.specialization_fingerprint


def test_later_bad_recipe_never_publishes_a_partial_freeze(authoring):
    _, recipe = add_future_operator(authoring)
    bad = replace(
        recipe,
        headers=(replace(recipe.headers[0], digest="0" * 64),) + recipe.headers[1:],
    )
    write_json(authoring.root / "csrc/recipes/future_matmul.json", [bad.to_dict()])
    before = (authoring.registry.read_bytes(), authoring.frozen.read_bytes())
    with pytest.raises(SchemaError, match="digest"):
        declarations.freeze_production_kernels(source_root=authoring.root)
    assert (authoring.registry.read_bytes(), authoring.frozen.read_bytes()) == before


def test_partial_selection_cannot_publish_missing_new_declaration(authoring):
    add_future_operator(authoring)
    before = authoring.frozen.read_bytes()
    with pytest.raises(SchemaError, match="freeze all newly registered"):
        declarations.freeze_production_kernels(
            source_root=authoring.root, kernel_ids=(authoring.recipe.kernel_id,)
        )
    assert authoring.frozen.read_bytes() == before


@pytest.mark.parametrize(
    "selection", [("missing/id",), (MATMUL_BIAS_SILU_KERNEL_ID,) * 2]
)
def test_invalid_selection_rejected_before_toolchain(authoring, monkeypatch, selection):
    monkeypatch.setattr(
        declarations.AscendCToolchain,
        "discover",
        lambda *a: pytest.fail("invalid selection reached discovery"),
    )
    with pytest.raises(SchemaError):
        declarations.freeze_production_kernels(
            source_root=authoring.root, kernel_ids=selection
        )


def test_freeze_rejects_target_mismatch_without_writing(authoring):
    before = authoring.frozen.read_bytes()
    with pytest.raises(SchemaError, match="requires target Ascend950"):
        declarations.freeze_production_kernels(
            source_root=authoring.root, targets=("Ascend910B1",)
        )
    assert authoring.frozen.read_bytes() == before


def test_duplicate_recipe_identity_is_rejected(authoring):
    shutil.copyfile(
        authoring.root / "csrc/recipes/matmul_bias_silu.json",
        authoring.root / "csrc/recipes/duplicate.json",
    )
    with pytest.raises(SchemaError, match="duplicate recipe kernel_id"):
        declarations.freeze_production_kernels(source_root=authoring.root)


def test_unselected_stale_capability_cannot_be_published(authoring):
    _, future = add_future_operator(authoring)
    payload = json.loads(authoring.registry.read_text())
    payload["kernels"][0]["priority"] += 1
    write_json(authoring.registry, payload)
    before = authoring.frozen.read_bytes()
    with pytest.raises(SchemaError, match="unselected declaration is stale"):
        declarations.freeze_production_kernels(
            source_root=authoring.root, kernel_ids=(future.kernel_id,)
        )
    assert authoring.frozen.read_bytes() == before


def test_dependency_names_are_data_and_shared_by_aot_and_jit(
    authoring, monkeypatch, tmp_path
):
    from flashinfer.ascendc import _dependency_roots

    dependencies = (
        BuildDependency("support", "a" * 40, "headers"),
        BuildDependency("another", "b" * 40),
    )
    recipe = replace(authoring.recipe, dependencies=dependencies)
    roots = {}
    for dependency in dependencies:
        root = tmp_path / dependency.name
        (root / dependency.include_dir).mkdir(parents=True)
        monkeypatch.setenv("FLASHINFER_%s_ROOT" % dependency.name.upper(), str(root))
        roots[dependency.name] = root
    assert dependency_roots(recipe, authoring.root) == roots
    assert _dependency_roots(recipe, authoring.root) == roots
    assert build._dependencies(recipe, authoring.root, require_explicit=True) == roots
    monkeypatch.delenv("FLASHINFER_SUPPORT_ROOT")
    assert (
        build._dependencies(
            recipe, authoring.root, dependency_roots=roots, require_explicit=True
        )
        == roots
    )
    with pytest.raises(SchemaError, match="missing dependency support"):
        build._dependencies(recipe, authoring.root, require_explicit=True)


def test_dependency_override_conflicts_fail_explicitly():
    with pytest.raises(SchemaError, match="conflicting"):
        build._dependency_overrides({"catlass": "/one"}, "/two")


def test_missing_dependency_does_not_read_native_checkout(authoring, monkeypatch):
    monkeypatch.setattr(
        Path, "is_dir", lambda *a: pytest.fail("native queried a dependency")
    )
    assert dependency_roots(authoring.recipe, authoring.root) == {}


def test_later_source_failure_prevents_all_compilation(
    authoring, monkeypatch, tmp_path
):
    from flashinfer import ascendc

    descriptor, recipe, spec = authoring.descriptor, authoring.recipe, authoring.spec
    next_id = recipe.kernel_id + "/next"
    contracts = {
        recipe.kernel_id: (descriptor, recipe, spec),
        next_id: (
            replace(descriptor, kernel_id=next_id),
            replace(recipe, kernel_id=next_id),
            replace(spec, kernel_id=next_id),
        ),
    }
    monkeypatch.setattr(ascendc, "_contracts", lambda: contracts)
    monkeypatch.setattr(
        build.AscendCToolchain,
        "discover",
        lambda *a: SimpleNamespace(fingerprint=spec.environment_fingerprint),
    )

    def resolve(actual, **kwargs):
        if actual.kernel_id == next_id:
            raise SchemaError("later source failed preflight")
        return SimpleNamespace(
            fingerprint=dict(spec.input_fingerprints)["resolved_recipe"]
        )

    monkeypatch.setattr(build, "resolve_build_recipe", resolve)
    monkeypatch.setattr(build, "AscendCBackend", lambda **kwargs: object())
    monkeypatch.setattr(
        build, "_publish", lambda *a: pytest.fail("partial batch compiled")
    )
    with pytest.raises(SchemaError, match="later source failed"):
        build.build_production_packages(output_dir=tmp_path / "release")


@pytest.mark.parametrize("with_output", [False, True])
def test_generic_cli_freezes_all_and_optionally_builds(
    authoring, monkeypatch, capsys, with_output
):
    arguments = ["flashinfer.aot", "--all", "--freeze"]
    if with_output:
        arguments += ["--output-dir", str(authoring.root / "release")]
    events = []
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(
        declarations,
        "freeze_production_kernels",
        lambda **kwargs: events.append(("freeze", kwargs))
        or ((authoring.descriptor, authoring.recipe, authoring.spec),),
    )
    monkeypatch.setattr(
        build,
        "build_production_packages",
        lambda **kwargs: events.append(("build", kwargs)),
    )
    build.main()
    assert [event for event, _ in events] == (
        ["freeze", "build"] if with_output else ["freeze"]
    )
    assert (
        json.loads(capsys.readouterr().out)["frozen"][0]["spec"]
        == authoring.spec.fingerprint
    )


def test_generic_cli_accepts_batch_selectors_and_dependency_roots(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "flashinfer.aot",
            "--kernel-id",
            "example/one",
            "--kernel-id",
            "example/two",
            "--dependency-root",
            "support=" + str(tmp_path),
            "--output-dir",
            str(tmp_path / "release"),
        ],
    )
    monkeypatch.setattr(
        build, "build_production_packages", lambda **kwargs: calls.append(kwargs)
    )
    build.main()
    assert calls[0]["kernel_ids"] == ["example/one", "example/two"]
    assert calls[0]["dependency_roots"] == {"support": tmp_path}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--all"],
        ["--freeze"],
        ["--freeze", "--config", "config.json"],
        ["--all", "--output-dir", "out", "--dependency-root", "invalid"],
        ["--all", "--output-dir", "out", "--dependency-root", "=path"],
        ["--all", "--output-dir", "out", "--dependency-root", "bad/name=path"],
        [
            "--all",
            "--output-dir",
            "out",
            "--dependency-root",
            "support=a",
            "--dependency-root",
            "support=b",
        ],
    ],
)
def test_generic_cli_rejects_ambiguous_or_incomplete_requests(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["flashinfer.aot", *arguments])
    with pytest.raises(SystemExit) as caught:
        build.main()
    assert caught.value.code == 2


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
def test_batch_cli_cold_build_cache_reuse_and_public_aot_jit(tmp_path):
    """Compile through the generic entry, then execute the unchanged facades."""
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.ascendc import _contracts

    assert torch.npu.is_available(), "batch compiler acceptance requires real NPU"
    assert torch.npu.device_count() == 8, "expected the declared eight Ascend devices"
    assert (
        Path(os.environ.get("ASCEND_HOME_PATH", "/absent")) / "bin/bisheng"
    ).is_file()
    catlass = Path(os.environ.get("FLASHINFER_CATLASS_ROOT", "/absent"))
    assert (
        catlass / "include"
    ).is_dir(), "set FLASHINFER_CATLASS_ROOT to the pinned checkout"
    environment = dict(os.environ)
    environment.pop("FLASHINFER_AOT_PATHS", None)
    environment.pop("FLASHINFER_NPU_JIT_DISABLE", None)
    commands = []

    def run(name, arguments, extra_env=None):
        command = [sys.executable, "-B", *arguments]
        commands.append(dict(name=name, command=command))
        write_json(tmp_path / "commands.json", commands)
        log_path = tmp_path / (name + ".log")
        with log_path.open("w") as output:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={**environment, **(extra_env or {})},
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        (tmp_path / (name + ".exit")).write_text(str(result.returncode) + "\n")
        output = log_path.read_text()
        assert result.returncode == 0, "%s: %s\n%s" % (name, log_path, output[-5000:])
        write_json(tmp_path / "progress.json", dict(completed=name))
        return output

    command = [
        "-m",
        "flashinfer.aot",
        "--all",
        "--output-dir",
        str(tmp_path / "release"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--dependency-root",
        "catlass=" + str(catlass),
    ]
    builds = []
    for name, expected_hit in (("aot-cold", False), ("aot-warm", True)):
        output = run(name, command)
        entries = [
            json.loads(line) for line in output.splitlines() if line.startswith('{"')
        ]
        packages = [entry for entry in entries if "kernel_id" in entry]
        assert {entry["kernel_id"] for entry in packages} == set(_contracts())
        assert all(entry["disk_cache_hit"] is expected_hit for entry in packages)
        assert entries[-1]["summary"] == dict(
            requested=len(_contracts()), built=len(_contracts()), failed=0, skipped=0
        )
        builds.append(packages)
    assert {entry["spec"] for entry in builds[0]} == {
        entry["spec"] for entry in builds[1]
    }

    aot = []
    jit = []
    for name, module in (
        ("matmul", "tests.support.matmul_bias_silu_acceptance"),
        ("attention", "tests.support.flash_attention_acceptance"),
    ):
        output = run(
            name + "-aot-numerical",
            [
                "-m",
                module,
                "numerical",
                str(tmp_path / (name + "-aot-store")),
            ],
            {"FLASHINFER_AOT_PATHS": str(tmp_path / "release")},
        )
        result = next(
            json.loads(line)
            for line in output.splitlines()
            if line.startswith('{"') and '"scenario"' in line
        )
        assert result["result"] == "PASS" and result["source"] == "aot"
        assert result["compiler_calls"] == result["toolchain_calls"] == 0
        assert len(result["errors"]) == (8 if name == "matmul" else 12)
        aot.append(result)
        output = run(
            name + "-cold-jit",
            [
                "-m",
                module,
                "missing_enabled",
                str(tmp_path / (name + "-jit-store")),
            ],
        )
        result = next(
            json.loads(line)
            for line in output.splitlines()
            if line.startswith('{"') and '"scenario"' in line
        )
        assert result["result"] == "PASS" and result["source"] == "jit"
        assert result["compiler_calls"] == result["toolchain_calls"] == 1
        assert result["repeated_public_call"] is True
        jit.append(result)
    write_json(
        tmp_path / "batch-results.json",
        dict(builds=builds, aot_numerical=aot, cold_jit=jit),
    )
