"""Wheel build modes must preserve complete releases and reject ambiguity."""

from pathlib import Path
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


@pytest.mark.parametrize("first,second", [
    (MODES[0], MODES[1]), (MODES[0], MODES[2]), (MODES[1], MODES[2]),
])
def test_aot_wheel_modes_are_mutually_exclusive(wheel_setup, monkeypatch, first, second):
    monkeypatch.setenv(first, "1" if first == MODES[0] else "/release-input")
    monkeypatch.setenv(second, "/release-input")
    with pytest.raises(ValueError, match="mutually exclusive"):
        wheel_setup["aot_requested"]()


@pytest.mark.parametrize("all_value", [None, "1"])
def test_all_wheel_build_exports_every_builder_output(wheel_setup, monkeypatch, tmp_path, all_value):
    from flashinfer.aot import build
    from flashinfer.aclnn import build as aclnn_build

    monkeypatch.chdir(ROOT)
    def native_build(output_dir):
        output_dir.mkdir(parents=True)
        for name in ("libflashinfer_aclnn.so", "manifest.json", "build.log"):
            (output_dir / name).write_text("fixture native adapter")

    monkeypatch.setattr(aclnn_build, "build_adapter", native_build)
    if all_value is not None:
        monkeypatch.setenv("FLASHINFER_AOT_ALL", all_value)
    monkeypatch.setenv("FLASHINFER_CATLASS_ROOT", "/explicit-catlass")
    monkeypatch.setenv("FLASHINFER_AOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(build_py, "run", lambda self: None)
    calls = []

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
    stale_operator = build_inputs / "csrc/kernels/ascendc/matmul_probe/native.cpp"
    stale_operator.parent.mkdir(parents=True)
    stale_operator.write_text("stale source before per-operator host/kernel grouping")
    command.run()
    destination = tmp_path / "build/flashinfer/data/op_packages"
    assert {path.name for path in destination.iterdir()} == {"a" * 64, "b" * 64}
    assert len(command._aot_outputs) == 2
    native = tmp_path / "build/flashinfer/data/aclnn"
    assert {path.name for path in native.iterdir()} == {"libflashinfer_aclnn.so", "manifest.json"}
    assert set(command._native_outputs) == {str(native / name) for name in
                                           ("libflashinfer_aclnn.so", "manifest.json")}
    assert (tmp_path / "aclnn-native-build.log").is_file()
    assert calls[0]["catlass_root"] == "/explicit-catlass"
    assert calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert distribution.has_ext_modules()
    assert (tmp_path / "build/flashinfer/data/build/csrc/recipes/flash_attention.json").is_file()
    assert not (build_inputs / "kernels").exists()
    assert not stale_operator.exists()
    for name in ("native.cpp", "catlass.cpp"):
        relative = Path("csrc/kernels/ascendc/matmul_probe/op_kernel") / name
        assert (build_inputs / relative).read_bytes() == (ROOT / relative).read_bytes()

    # A later source-only build in the same directory must not retain AOT bytes.
    monkeypatch.setenv("FLASHINFER_AOT_ALL", "0")
    command.run()
    assert not destination.exists()
    assert command._aot_outputs == []
    assert not native.exists()
    assert command._native_outputs == []
    assert not distribution.has_ext_modules()


def test_default_wheel_requests_all_production_packages(wheel_setup):
    assert wheel_setup["aot_mode"]() == "FLASHINFER_AOT_ALL"
    assert wheel_setup["AOTDistribution"]().has_ext_modules()


@pytest.mark.parametrize("value", ["0", ""])
def test_source_only_wheel_is_explicit(wheel_setup, monkeypatch, value):
    monkeypatch.setenv("FLASHINFER_AOT_ALL", value)
    assert wheel_setup["aot_mode"]() is None
    assert not wheel_setup["AOTDistribution"]().has_ext_modules()


def test_default_catlass_checkout_is_a_build_input(wheel_setup, monkeypatch, tmp_path):
    monkeypatch.delenv("FLASHINFER_CATLASS_ROOT", raising=False)
    function = wheel_setup["build_catlass_root"]
    monkeypatch.setitem(function.__globals__, "__file__", str(tmp_path / "project/setup.py"))
    catlass = tmp_path / "catlass-ref"
    (catlass / "include").mkdir(parents=True)
    assert function() == str(catlass)
