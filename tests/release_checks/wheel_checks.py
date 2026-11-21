"""Wheel build modes must preserve complete releases and reject ambiguity."""

from pathlib import Path
import json
import runpy

import pytest
import setuptools
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parents[2]
MODES = ("FLASHINFER_AOT_ALL", "FLASHINFER_AOT_CONFIG", "FLASHINFER_AOT_PREBUILT")


@pytest.fixture
def wheel_setup(monkeypatch):
    for name in MODES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    return runpy.run_path(str(ROOT / "setup.py"))


def test_author_directory_build_inputs_include_auxiliary_headers_and_local_cmake_data(
    wheel_setup, tmp_path, monkeypatch
):
    folder = tmp_path / "csrc/kernels/example"
    folder.mkdir(parents=True)
    for name in (
        "operator.py",
        "cases.py",
        "config.txt",
        "template.hpp",
        "CMakeLists.txt",
    ):
        (folder / name).write_text("fixture input\n")
    cache = folder / "__pycache__"
    cache.mkdir()
    (cache / "operator.pyc").write_bytes(b"cache")
    monkeypatch.chdir(tmp_path)
    assert {p.name for p in wheel_setup["build_inputs"]()} == {
        "operator.py",
        "cases.py",
        "config.txt",
        "template.hpp",
        "CMakeLists.txt",
    }


def test_wheel_rebuild_removes_obsolete_python_modules(
    wheel_setup, tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("FLASHINFER_AOT_ALL", "0")
    package = tmp_path / "build/flashinfer"
    package.mkdir(parents=True)
    (package / "removed_operator.py").write_text("stale module\n")

    def copy_current(command):
        package.mkdir(parents=True)
        (package / "current.py").write_text("current module\n")

    monkeypatch.setattr(build_py, "run", copy_current)
    command = wheel_setup["BuildPy"](wheel_setup["AOTDistribution"]())
    command.build_lib = str(tmp_path / "build")
    command.run()
    assert not (package / "removed_operator.py").exists()
    assert (package / "current.py").is_file()


def test_wheel_cleanup_cannot_remove_the_source_tree(wheel_setup, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("FLASHINFER_AOT_ALL", "0")
    monkeypatch.setattr(
        build_py, "run", lambda _: pytest.fail("must reject before copying")
    )
    command = wheel_setup["BuildPy"](wheel_setup["AOTDistribution"]())
    command.build_lib = str(ROOT)
    with pytest.raises(ValueError, match="overlap the package source"):
        command.run()


@pytest.mark.parametrize(
    "first,second",
    [
        (MODES[0], MODES[1]),
        (MODES[0], MODES[2]),
        (MODES[1], MODES[2]),
    ],
)
def test_aot_wheel_modes_are_mutually_exclusive(
    wheel_setup, monkeypatch, first, second
):
    monkeypatch.setenv(first, "1" if first == MODES[0] else "/release-input")
    monkeypatch.setenv(second, "/release-input")
    with pytest.raises(ValueError, match="mutually exclusive"):
        wheel_setup["aot_requested"]()


@pytest.mark.parametrize("all_value", [None, "1"])
def test_all_wheel_build_exports_every_builder_output(
    wheel_setup, monkeypatch, tmp_path, all_value
):
    from flashinfer.aot import build
    from flashinfer.aot import declarations
    from flashinfer.aclnn import build as aclnn_build

    monkeypatch.chdir(ROOT)

    def native_build(output_dir):
        output_dir.mkdir(parents=True)
        for name in ("libflashinfer_aclnn.so", "manifest.json", "build.log"):
            (output_dir / name).write_text("fixture native adapter")

    monkeypatch.setattr(aclnn_build, "build_adapter", native_build)
    monkeypatch.setenv("FLASHINFER_AOT_SOCS", "Ascend950")
    if all_value is not None:
        monkeypatch.setenv("FLASHINFER_AOT_ALL", all_value)
    monkeypatch.setenv("FLASHINFER_CATLASS_ROOT", "/explicit-catlass")
    monkeypatch.setenv("FLASHINFER_AOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(build_py, "run", lambda self: None)
    calls = []
    freeze_calls = []
    monkeypatch.setattr(
        declarations,
        "freeze_production_kernels",
        lambda **options: freeze_calls.append(options),
    )

    def build_all(**options):
        calls.append(options)
        for spec in ("a" * 64, "b" * 64):
            package = options["output_dir"] / spec
            package.mkdir(parents=True)
            (package / "manifest.json").write_text("fixture release")

    monkeypatch.setattr(build, "build_production_packages", build_all)
    distribution = wheel_setup["AOTDistribution"]()
    command = wheel_setup["BuildPy"](distribution)
    command.build_lib = str(tmp_path / "build")
    build_inputs = tmp_path / "build/flashinfer/data/build"
    stale_source = build_inputs / "kernels/ascendc/matmul_probe/native.cpp"
    stale_source.parent.mkdir(parents=True)
    stale_source.write_text("stale source from the previous layout")
    stale_operator = build_inputs / "csrc/kernels/gemm/matmul_probe/native.cpp"
    stale_operator.parent.mkdir(parents=True)
    stale_operator.write_text("stale source before per-operator host/kernel grouping")
    command.run()
    destination = tmp_path / "build/flashinfer/data/op_packages"
    assert {path.name for path in destination.iterdir()} == {"a" * 64, "b" * 64}
    assert len(command._aot_outputs) == 2
    native = tmp_path / "build/flashinfer/data/aclnn"
    assert {path.name for path in native.iterdir()} == {
        "libflashinfer_aclnn.so",
        "manifest.json",
    }
    assert set(command._native_outputs) == {
        str(native / name) for name in ("libflashinfer_aclnn.so", "manifest.json")
    }
    assert (tmp_path / "aclnn-native-build.log").is_file()
    assert calls[0]["catlass_root"] == "/explicit-catlass"
    assert freeze_calls == [{"dependency_roots": {"catlass": "/explicit-catlass"}}]
    assert calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert distribution.has_ext_modules()
    assert (
        tmp_path / "build/flashinfer/data/build/csrc/recipes/flash_attention.json"
    ).is_file()
    assert not (build_inputs / "kernels").exists()
    assert not stale_operator.exists()
    for name in ("native.cpp", "catlass.cpp"):
        relative = Path("csrc/kernels/gemm/matmul_probe/op_kernel") / name
        assert (build_inputs / relative).read_bytes() == (ROOT / relative).read_bytes()

    # A later source-only build in the same directory must not retain AOT bytes.
    monkeypatch.setenv("FLASHINFER_AOT_ALL", "0")
    command.run()
    assert not destination.exists()
    assert command._aot_outputs == []
    assert not native.exists()
    assert command._native_outputs == []
    assert not distribution.has_ext_modules()


def test_default_wheel_requests_exactly_the_local_chip(wheel_setup, monkeypatch):
    from types import SimpleNamespace
    from flashinfer import _wheel
    monkeypatch.delenv("FLASHINFER_AOT_SOCS", raising=False)
    monkeypatch.setattr(_wheel, "current_hardware", lambda: SimpleNamespace(soc_model="Ascend910B"))
    assert wheel_setup["aot_mode"]() == "FLASHINFER_AOT_ALL"
    assert _wheel.wheel_target() == "a2a3"
    assert _wheel.distribution_name("Ascend910B") == "flashinfer_ascend910b"
    assert wheel_setup["AOTDistribution"]().has_ext_modules()


@pytest.mark.parametrize("selector", ["flash_attention", "all"])
def test_config_selects_implementations_within_the_wheel_chip(
    wheel_setup, monkeypatch, tmp_path, selector
):
    from flashinfer.aot import build, declarations
    from flashinfer.aclnn import build as aclnn_build
    from flashinfer.ascendc import _contracts
    from flashinfer.jit import cache
    from tests.production_inventory import PRODUCTION

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("FLASHINFER_AOT_SOCS", "Ascend910B")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(recipe_id=selector, implementation="catlass",
                                     target_soc="Ascend950", output_kind="op_package")))
    monkeypatch.setenv("FLASHINFER_AOT_CONFIG", str(config))
    monkeypatch.setattr(build_py, "run", lambda _: None)
    monkeypatch.setattr(declarations, "freeze_production_kernels", lambda **kwargs: None)
    monkeypatch.setattr(aclnn_build, "build_adapter", lambda output: output.mkdir(parents=True))
    monkeypatch.setattr(cache, "verify_build_directory", lambda _: None)
    calls = []

    def publish(kernel_ids, output_dir):
        for kernel_id in kernel_ids:
            spec = _contracts()[kernel_id][2]
            destination = output_dir / spec.fingerprint
            destination.mkdir(parents=True)
            (destination / "manifest.json").write_text(json.dumps(dict(spec=spec.to_dict())))

    def compile_config(value, **options):
        pytest.fail("wheel CONFIG must select exact frozen variants across its family")

    def compile_all(**options):
        calls.append("a2a3")
        assert set(options["kernel_ids"]) == {row[0] for row in PRODUCTION
            if row[-1] in ("Ascend910B", "Ascend910C")
            and (selector == "all" or row[2] == "flash_attention")}
        publish(options["kernel_ids"], options["output_dir"])

    monkeypatch.setattr(build, "compile_and_package", compile_config)
    monkeypatch.setattr(build, "build_production_packages", compile_all)
    command = wheel_setup["BuildPy"](wheel_setup["AOTDistribution"]())
    command.build_lib = str(tmp_path / "build")
    command.run()
    assert calls == ["a2a3"]
    metadata = json.loads((tmp_path / "build/flashinfer/data/wheel.json").read_text())
    assert metadata["family"] == "a2a3"
    assert metadata["soc_models"] == ["Ascend910B", "Ascend910C"]
    assert len(metadata["kernel_ids"]) == (2 if selector == "flash_attention" else 7)


@pytest.mark.parametrize("value", ["0", ""])
def test_source_only_wheel_is_explicit(wheel_setup, monkeypatch, value):
    monkeypatch.setenv("FLASHINFER_AOT_ALL", value)
    assert wheel_setup["aot_mode"]() is None
    assert not wheel_setup["AOTDistribution"]().has_ext_modules()


def test_default_catlass_checkout_is_a_build_input(wheel_setup, monkeypatch, tmp_path):
    monkeypatch.delenv("FLASHINFER_CATLASS_ROOT", raising=False)
    function = wheel_setup["build_catlass_root"]
    monkeypatch.setitem(
        function.__globals__, "__file__", str(tmp_path / "project/setup.py")
    )
    catlass = tmp_path / "catlass-ref"
    (catlass / "include").mkdir(parents=True)
    assert function() == str(catlass)
