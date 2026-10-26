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


def test_all_wheel_build_exports_every_builder_output(wheel_setup, monkeypatch, tmp_path):
    from flashinfer.aot import build

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("FLASHINFER_AOT_ALL", "1")
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
    command.run()
    destination = tmp_path / "build/flashinfer/data/op_packages"
    assert {path.name for path in destination.iterdir()} == {"a" * 64, "b" * 64}
    assert len(command._aot_outputs) == 2
    assert calls[0]["catlass_root"] == "/explicit-catlass"
    assert calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert distribution.has_ext_modules()
    assert (tmp_path / "build/flashinfer/data/build/csrc/recipes/flash_attention.json").is_file()

    # A later source-only build in the same directory must not retain AOT bytes.
    monkeypatch.delenv("FLASHINFER_AOT_ALL")
    command.run()
    assert not destination.exists()
    assert command._aot_outputs == []
    assert not distribution.has_ext_modules()
