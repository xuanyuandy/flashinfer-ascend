"""Author directories drive generation, while frozen identities keep their guards."""

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer.operators.authoring import discover, generate
from flashinfer.operators.catalog import definition
from flashinfer.operators.definition import schema_arguments
from flashinfer.jit.recipe import BuildRecipe, resolve_build_recipe
from flashinfer.runtime import SchemaError
from flashinfer._resources import data_root, recipe_root

ROOT = Path(__file__).resolve().parents[1]
NAME = "grouped_matmul_swiglu_quant"
KERNEL_ID = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    data = root / "flashinfer/data/kernels"
    data.mkdir(parents=True)
    for file in ("registry.json", "ascendc_specs.json"):
        shutil.copyfile(data_root(ROOT) / "kernels" / file, data / file)
    paths = {
        "native": root / "csrc/kernels/gemm/local_probe",
        "logical": root / "flashinfer/operators/gemm/local_probe",
        "implementation": root / "flashinfer/ascendc/gemm/local_probe",
    }
    shutil.copytree(ROOT / "csrc/kernels/gemm" / NAME, paths["native"])
    shutil.copytree(ROOT / "flashinfer/operators/gemm" / NAME, paths["logical"])
    shutil.copytree(ROOT / "flashinfer/ascendc/gemm" / NAME, paths["implementation"])
    for directory in (paths["logical"], paths["implementation"]):
        for source in directory.rglob("*.py"):
            text = source.read_text().replace(
                "gemm.grouped_matmul_swiglu_quant", "gemm.local_probe"
            )
            text = text.replace("gemm/grouped_matmul_swiglu_quant", "gemm/local_probe")
            text = text.replace(
                'name="grouped_matmul_swiglu_quant"', 'name="local_probe"'
            )
            text = text.replace(
                "name='grouped_matmul_swiglu_quant'", "name='local_probe'"
            )
            text = text.replace(
                "recipe_id='grouped_matmul_swiglu_quant'", "recipe_id='local_probe'"
            )
            text = text.replace('recipe_id="grouped_matmul_swiglu_quant"', 'recipe_id="local_probe"')
            text = text.replace(
                "flashinfer.operators.gemm.local_probe.operator",
                "flashinfer.operators.gemm.grouped_matmul_swiglu_quant.operator",
            )
            if source.name == "implementation.py":
                text = text.replace(
                    "from .runner import make_runner, admission",
                    "from .runner import make_runner, admission\nfrom dataclasses import replace\n"
                    "OPERATOR = replace(OPERATOR, name='local_probe', op='gemm.local_probe')",
                )
            source.write_text(text)
    (root / "csrc/recipes").mkdir()
    shutil.copyfile(ROOT / "csrc/CMakeLists.txt", root / "csrc/CMakeLists.txt")
    shutil.copytree(ROOT / "csrc/cmake", root / "csrc/cmake")
    return root, paths


def recipe(root):
    return BuildRecipe.from_dict(
        json.loads((root / "csrc/recipes/local_probe.json").read_text())[0]
    )


def test_new_directory_generates_all_glue_without_changing_framework_sources(project):
    root, directory = project
    before = json.loads((root / "flashinfer/data/kernels/registry.json").read_text())
    selected = generate(source_root=root)
    assert selected == ("ascendc/gemm/local_probe/native/op_package/Ascend910B",)
    after = json.loads((root / "flashinfer/data/kernels/registry.json").read_text())
    assert after["kernels"][:-1] == before["kernels"]
    assert after["kernels"][-1]["op"] == "gemm.local_probe"
    catalog = json.loads((root / "flashinfer/data/operators.json").read_text())
    assert [(r["name"], r["domain"]) for r in catalog["operators"]] == [
        ("local_probe", "gemm")
    ]
    assert (directory["logical"] / "operator.py").is_file()
    assert (directory["logical"] / "cases.py").is_file()
    implementations = json.loads(
        (root / "flashinfer/data/implementations.json").read_text()
    )
    assert [r["module"] for r in implementations["implementations"]] == [
        "flashinfer.ascendc.gemm.local_probe.implementation"
    ]
    assert not (root / "flashinfer/operators/_generated").exists()
    generate(source_root=root, check=True)


@pytest.mark.parametrize(
    "owner,path",
    [
        ("native", "CMakeLists.txt"),
        ("native", "settings.cmake"),
        ("logical", "operator.py"),
        ("logical", "cases.py"),
        ("implementation", "runner.py"),
        ("native", "op_kernel/arch22/grouped_matmul_swiglu_quant.h"),
        ("native", "op_kernel/extra.hpp"),
    ],
)
def test_all_author_inputs_invalidate_recipe_and_stale_files_are_detected(
    project, owner, path
):
    root, directories = project
    directory = directories[owner]
    if path == "settings.cmake":
        (directory / path).write_text("# declared local CMake include\n")
    if path == "op_kernel/extra.hpp":
        (directory / path).write_text("// auxiliary template header\n")
    generate(source_root=root)
    original = recipe(root)
    target = directory / path
    target.write_text(
        target.read_text()
        + ("\n// changed\n" if target.suffix == ".h" else "\n# changed\n")
    )
    with pytest.raises(SchemaError, match="stale"):
        generate(source_root=root, check=True)
    with pytest.raises(SchemaError, match="digest mismatch"):
        resolve_build_recipe(original, source_root=root, toolchain_fingerprint="a" * 64)
    generate(source_root=root)
    changed = recipe(root)
    assert changed.fingerprint != original.fingerprint
    assert (
        changed.source == original.source
    )  # A header/CMake/hook change still matters.


def test_generation_is_idempotent_and_does_not_touch_unselected_frozen_specs(project):
    root, _ = project
    frozen = root / "flashinfer/data/kernels/ascendc_specs.json"
    old = frozen.read_bytes()
    generate(source_root=root)
    paths = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    generate(source_root=root)
    assert all((p.stat().st_mtime_ns, p.read_bytes()) == before[p] for p in paths)
    assert frozen.read_bytes() == old


def test_missing_case_provider_fails_before_publishing_any_record(project):
    root, directory = project
    registry = root / "flashinfer/data/kernels/registry.json"
    old = registry.read_bytes()
    (directory["logical"] / "cases.py").unlink()
    with pytest.raises(SchemaError, match="cases.py"):
        generate(source_root=root)
    assert registry.read_bytes() == old
    assert not (root / "csrc/recipes/local_probe.json").exists()


def test_removing_the_last_author_directory_cannot_report_a_fresh_catalog(project):
    root, directory = project
    generate(source_root=root)
    (directory["logical"] / "operator.py").unlink()
    with pytest.raises(SchemaError, match="managed operator source was removed"):
        generate(source_root=root, check=True)


def test_symlink_source_is_not_silently_included_in_a_closed_build(project, tmp_path):
    root, directory = project
    external = tmp_path / "external.h"
    external.write_text("// untracked dependency\n")
    (directory["native"] / "op_kernel/external.h").symlink_to(external)
    with pytest.raises(SchemaError, match="in-directory regular"):
        generate(source_root=root)


def test_author_directory_cannot_take_over_an_existing_production_identity(project):
    root, directory = project
    registry = root / "flashinfer/data/kernels/registry.json"
    before = registry.read_bytes()
    source = directory["implementation"] / "implementation.py"
    source.write_text(
        source.read_text().replace(
            "ascendc/gemm/local_probe/native/op_package/Ascend910B",
            "ascendc/gemm/matmul/native/op_package/Ascend950",
        )
    )
    with pytest.raises(SchemaError, match="unrelated production"):
        generate(source_root=root)
    assert registry.read_bytes() == before


def test_optional_cmake_and_nested_python_helpers_are_collected(project):
    root, directory = project
    source = directory["implementation"] / "implementation.py"
    source.write_text(
        source.read_text().replace('cmake="CMakeLists.txt"', "cmake=None")
    )
    (directory["native"] / "CMakeLists.txt").unlink()
    helpers = directory["implementation"] / "helpers"
    helpers.mkdir()
    (helpers / "__init__.py").write_text('"""Author helper package."""\n')
    (helpers / "values.py").write_text("VALUE = 1\n")
    generate(source_root=root)
    assert recipe(root).cmake_entry is None
    author_helper = directory["implementation"] / "helpers/values.py"
    assert author_helper.read_text() == "VALUE = 1\n"
    assert author_helper.relative_to(root).as_posix() in {
        r.locator for r in recipe(root).extra_build_files
    }
    resolved = resolve_build_recipe(
        recipe(root), source_root=root, toolchain_fingerprint="a" * 64
    )
    assert "csrc/cmake/operator/CMakeLists.txt" in dict(resolved.build_files)


def test_public_integration_points_have_no_operator_specific_glue():
    for name in ("__init__.py", "gemm.py", "aclnn/_ops.py", "selftest.py"):
        assert NAME not in (ROOT / "flashinfer" / name).read_text(), name


def test_extra_build_contract_is_opt_in_and_roundtrips_without_changing_legacy_records():
    from tests.legacy_ascendc import ROOT as LEGACY_ROOT, load_build_recipes

    raw = json.loads((LEGACY_ROOT / "csrc/recipes/matmul_probe.json").read_text())
    for declaration, original in zip(
        load_build_recipes(LEGACY_ROOT / "csrc/recipes/matmul_probe.json"), raw
    ):
        assert json.loads(json.dumps(declaration.to_dict())) == original
        with pytest.raises(SchemaError, match="hashed extra"):
            replace(declaration, cmake_entry="untracked.cmake")
    current = BuildRecipe.from_dict(
        json.loads((recipe_root(ROOT) / (NAME + ".json")).read_text())[0]
    )
    assert BuildRecipe.from_dict(current.to_dict()) == current
    assert current.cmake_entry.endswith("/CMakeLists.txt")
    assert current.extra_build_files


def test_api_and_torch_schemas_derive_from_the_same_author_declaration():
    import flashinfer

    operator = definition(NAME)
    assert (
        inspect.signature(flashinfer.grouped_matmul_swiglu_quant)
        == operator.public_signature
    )
    assert (
        operator.public_signature.parameters["bias"].kind
        == inspect.Parameter.KEYWORD_ONLY
    )
    schema, _ = operator.torch_schema()
    assert schema.index("Tensor? bias") < schema.index("*")
    out_schema, _ = operator.torch_schema(out=True)
    assert "Tensor(a!) output" in out_schema and "Tensor(b!) output_scale" in out_schema


def test_generic_fake_supports_a_different_operator_with_a_single_output(monkeypatch):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flashinfer.operators import Invocation, OutputSpec
    from flashinfer.operators import runtime
    from flashinfer.jit.recipe import ShapeConstraint, TensorParameter

    original = definition(NAME)
    tensors = (
        TensorParameter("x", "input", "float32", ("M",)),
        TensorParameter("y", "output", "float32", ("M",)),
    )
    operator = replace(
        original,
        name="single_output_probe",
        schema="(Tensor x) -> Tensor",
        output_names=("y",),
        op="gemm.single_output_probe",
        validate_values=None,
        infer=lambda a, layouts: Invocation(
            (a["x"],),
            (OutputSpec(tuple(a["x"].shape), "float32", a["x"].device),),
            {},
            None,
        ),
    )
    monkeypatch.setattr(runtime, "definition", lambda name: operator)
    monkeypatch.setattr(
        "flashinfer.ascendc.load_contract",
        lambda _: pytest.fail("fake cannot read a native contract"),
    )
    monkeypatch.setattr(
        runtime, "dispatch", lambda *a, **k: pytest.fail("fake cannot dispatch")
    )
    with FakeTensorMode():
        x = torch.empty(7)
        result = runtime.fake(operator.name, operator.bind(x))
        assert result.shape == (7,) and result.dtype == torch.float32
        assert runtime.fake(operator.name, operator.bind(x, out=result)) is None


def test_import_and_source_generation_work_without_torch_or_sdk():
    code = """
import sys
sys.modules['torch'] = None
sys.modules['torch_npu'] = None
import flashinfer
from flashinfer.operators.authoring import generate
assert callable(flashinfer.grouped_matmul_swiglu_quant)
assert generate(check=True)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
