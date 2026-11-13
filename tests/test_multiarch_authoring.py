"""One operator/runner, distinct compiled SoCs, vLLM-style device source layout."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from flashinfer._resources import data_root, recipe_root
from flashinfer.ascendc.definition import AscendCImplementation
from flashinfer.operators import Support
from flashinfer.operators.authoring import discover_implementations, generate
from flashinfer.jit import load_build_recipes, resolve_build_recipe
from flashinfer.runtime import (
    HardwareSignature,
    KernelRegistry,
    SchemaError,
    kernel_registry_scope,
)
from tests.multiarch_case import OPERATOR, SOCS, make_runner, project

IDS = {
    "ascendc/gemm/multiarch_probe/native/Ascend910B",
    "ascendc/gemm/multiarch_probe/native/Ascend910C",
    "ascendc/gemm/multiarch_probe/native/Ascend950",
}
ARCHITECTURES = {
    "Ascend910B": "dav-2201",
    "Ascend910C": "dav-2201",
    "Ascend950": "dav-3510",
}


@pytest.fixture
def source(tmp_path):
    return project(tmp_path / "project")


def test_all_target_variants_generate_once_with_separate_recipes_and_one_abi(source):
    assert set(generate(source_root=source)) == IDS
    declarations = discover_implementations(source)
    assert len(declarations) == 3
    assert {d.runner_factory for _, _, d in declarations} == {make_runner}
    assert {p.name for p in recipe_root(source).glob("*.json")} == {
        "multiarch_probe__" + target + ".json" for target in SOCS
    }
    records = json.loads((data_root(source) / "kernels/registry.json").read_text())[
        "kernels"
    ]
    assert {r["kernel_id"] for r in records} == IDS
    assert len({json.dumps(r["launch_abi"], sort_keys=True) for r in records}) == 1
    for path in recipe_root(source).glob("*.json"):
        (recipe,) = load_build_recipes(path)
        assert recipe.npu_arch == ARCHITECTURES[recipe.target_soc]
        assert recipe.recipe_id == "multiarch_probe"
        assert {"arch22", "arch35"} == {
            Path(r.locator).parent.name
            for r in recipe.headers
            if r.locator.endswith("/compute.h")
        }
    generate(source_root=source, check=True)


def test_selected_variant_is_not_lost_behind_the_first_module_declaration(source):
    selected = "ascendc/gemm/multiarch_probe/native/Ascend910C"
    assert generate(source_root=source, kernel_ids=(selected,)) == (selected,)
    assert {p.name for p in recipe_root(source).glob("*.json")} == {
        "multiarch_probe__Ascend910C.json"
    }
    records = json.loads((data_root(source) / "kernels/registry.json").read_text())[
        "kernels"
    ]
    assert [r["kernel_id"] for r in records] == [selected]


def test_architecture_headers_participate_in_closed_build_identity(source):
    generate(source_root=source)
    before = {
        r.target_soc: r
        for p in recipe_root(source).glob("*.json")
        for r in load_build_recipes(p)
    }
    path = source / "csrc/kernels/gemm/multiarch_probe/op_kernel/arch35/compute.h"
    path.write_text(path.read_text() + "\n// changed architecture implementation\n")
    with pytest.raises(SchemaError, match="stale"):
        generate(source_root=source, check=True)
    with pytest.raises(SchemaError, match="digest mismatch"):
        resolve_build_recipe(
            before["Ascend950"], source_root=source, toolchain_fingerprint="1" * 64
        )
    generate(source_root=source)
    after = {
        r.target_soc: r
        for p in recipe_root(source).glob("*.json")
        for r in load_build_recipes(p)
    }
    assert set(after) == set(before) == set(SOCS)
    assert all(after[soc].fingerprint != before[soc].fingerprint for soc in SOCS)


@pytest.mark.parametrize(
    "models", [(), ("Ascend910B", "Ascend910B"), ("ascend910b1",), ("unknown",)]
)
def test_multi_target_declarations_reject_missing_duplicate_and_noncanonical_models(
    models,
):
    with pytest.raises(SchemaError):
        AscendCImplementation.for_targets(
            operator=OPERATOR, support=Support(soc=models)
        )


@pytest.mark.parametrize("soc", SOCS)
def test_common_logical_interface_selects_exact_target_before_using_shared_runner(
    source, monkeypatch, soc
):
    import torch
    import importlib
    from flashinfer.operators import runtime

    # This unit test injects hardware/layout metadata; native execution is below.
    signatures = importlib.import_module("flashinfer.runtime.signature")
    monkeypatch.setattr(signatures, "_tensor_physical_layout", lambda _: "ND")
    declarations = discover_implementations(source)
    calls = []

    def shared_runner(a, b, out):
        calls.append(True)
        torch.mm(a, b, out=out)

    descriptors = [
        replace(d.prepare_outputs(source)["descriptor"], runner=shared_runner)
        for _, _, d in declarations
    ]
    registry = KernelRegistry(descriptors)
    registry.freeze()
    monkeypatch.setattr(runtime, "definition", lambda _: OPERATOR)
    monkeypatch.setattr(runtime, "current_hardware", lambda: HardwareSignature(soc, 20))
    from flashinfer.operators import implementation, testing
    from flashinfer.operators.implementation import ImplementationDefinition

    candidates = tuple(
        ImplementationDefinition(d.kernel_id, d.op, d.backend, lambda d=d: d)
        for d in descriptors
    )
    monkeypatch.setattr(implementation, "implementations", lambda **_: candidates)
    monkeypatch.setattr(testing, "definition", lambda _: OPERATOR)
    monkeypatch.setattr(
        "flashinfer.runtime.current_hardware", lambda: HardwareSignature(soc, 20)
    )
    assert tuple(
        d.kernel_id for d in testing.target_aot_implementations(OPERATOR.op)
    ) == ("ascendc/gemm/multiarch_probe/native/" + soc,)
    a, b = torch.randn(16, 16), torch.randn(16, 16)
    with kernel_registry_scope(registry):
        actual = runtime.execute(OPERATOR.op, OPERATOR.bind(a, b))
        selected = next(d for d in descriptors if d.target_soc == soc)
        for other in descriptors:
            if other is selected:
                continue
            with pytest.raises(RuntimeError, match="target SoC"):
                runtime.execute(
                    OPERATOR.op, OPERATOR.bind(a, b, kernel_id=other.kernel_id)
                )
    assert calls == [True]
    torch.testing.assert_close(actual, a @ b)


@pytest.mark.real_compile
def test_three_targets_compile_and_one_runner_executes_on_available_real_device(
    source, tmp_path
):
    from flashinfer.aot.declarations import freeze_production_kernels
    from flashinfer.aot.build import export_op_package
    from flashinfer.jit import AscendCBackend, AscendCToolchain, DiskJitBuildStore
    from flashinfer.jit.build import _build_record
    from flashinfer.aclnn.build import build_adapter

    generate(source_root=source)
    contracts = freeze_production_kernels(source_root=source)
    assert {d.kernel_id for d, _, _ in contracts} == IDS
    toolchain = AscendCToolchain.discover()
    store = DiskJitBuildStore(
        os.environ.get("FLASHINFER_MULTIARCH_CACHE_DIR", tmp_path / "cache")
    )
    release = tmp_path / "release"
    evidence = []
    for descriptor, recipe, spec in contracts:
        resolved = resolve_build_recipe(
            recipe, source_root=source, toolchain_fingerprint=toolchain.fingerprint
        )
        backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
        record = _build_record(spec, backend=backend, store=store)
        path = export_op_package(store.directory(record), release)
        evidence.append(
            dict(
                soc=recipe.target_soc,
                arch=recipe.npu_arch,
                spec=spec.fingerprint,
                package=str(path),
            )
        )
    assert {row["soc"] for row in evidence} == set(SOCS)
    adapter = os.environ.get("FLASHINFER_OPERATOR_TEST_ADAPTER")
    if adapter is None:
        adapter = tmp_path / "adapter"
        build_adapter(adapter)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.multiarch_case",
            str(source),
            str(release),
            str(adapter),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "runtime.log").write_text(result.stdout + result.stderr)
    (tmp_path / "compiled-targets.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"runtime_compilations": 0' in result.stdout
    assert '"shared_runner": true' in result.stdout
    print(json.dumps(evidence))
    print(result.stdout)
