"""CANN path aliases retain verified bytes and release failed activation FDs."""

import ctypes as C
import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer.jit._legacy_recipe import export_op_package
from flashinfer.jit import op_package as package
from tests.test_ascendc_op_package import generation, unit  # noqa: F401


@pytest.fixture
def isolated_runtime(monkeypatch, unit):
    recipe, _, _ = unit
    calls = []

    def workspace(*args):
        return 0

    def launch(*args):
        return 0

    api = SimpleNamespace(**{recipe.host_entry: launch,
                            recipe.host_entry + "GetWorkspaceSize": workspace})

    def cdll(path, mode):
        calls.append(Path(path))
        return api

    monkeypatch.setattr(C, "CDLL", cdll)
    monkeypatch.setattr(package, "_active_package", None)
    # Track an initially absent key before the loader mutates os.environ.
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "")
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)
    try:
        yield calls, api
    finally:
        active = package._active_package
        package._active_package = None
        if active is not None and active[1]._registration_fd is not None:
            os.close(active[1]._registration_fd)


def test_dotted_store_uses_same_verified_directory_via_retained_fd(generation, isolated_runtime):
    store, record, paths = generation
    directory = store.directory(record)
    assert ".objects" in directory.parts
    module = package.load_op_package(directory, target_soc=record.artifact.target_soc)
    registration = Path(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    assert registration == module.registration_path
    assert "." not in str(registration)
    assert registration.name == module.vendor_path.name
    assert registration.samefile(module.vendor_path)
    descriptor = os.fstat(module._registration_fd)
    verified = directory.stat()
    assert (descriptor.st_dev, descriptor.st_ino) == (verified.st_dev, verified.st_ino)
    assert registration.parts[:4] == ("/", "proc", str(os.getpid()), "fd")
    assert (registration / "op_api/lib/libcust_opapi.so").read_bytes() == (
        directory / paths["api"]).read_bytes()
    assert package.load_op_package(directory, target_soc=record.artifact.target_soc) is module
    assert len(isolated_runtime[0]) == 1
    assert os.fstat(module._registration_fd).st_ino == verified.st_ino


@pytest.mark.parametrize("failure", ["dlopen", "symbol"])
@pytest.mark.parametrize("previous", [None, "/prior/vendor"])
def test_failed_activation_closes_alias_fd_and_restores_environment(
    generation, isolated_runtime, monkeypatch, failure, previous,
):
    store, record, _ = generation
    if previous is not None:
        monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", previous)
    captured = []
    original = OSError("fixture dlopen rejection")

    def cdll(path, mode):
        parts = os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
        registration = Path(parts[0])
        fd = int(registration.parts[4])
        assert os.fstat(fd).st_ino == store.directory(record).stat().st_ino
        assert parts[1:] == ([] if previous is None else [previous])
        captured.append(fd)
        if failure == "dlopen":
            raise original
        return SimpleNamespace()

    monkeypatch.setattr(C, "CDLL", cdll)
    error = OSError if failure == "dlopen" else AttributeError
    with pytest.raises(error) as failed:
        package.load_op_package(store.directory(record), target_soc=record.artifact.target_soc)
    if failure == "dlopen":
        assert failed.value is original
    assert len(captured) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(captured[0])
    assert closed.value.errno == errno.EBADF
    assert os.environ.get("ASCEND_CUSTOM_OPP_PATH") == previous
    assert package._active_package is None


def test_release_without_dots_keeps_original_registration_path(
    generation, isolated_runtime, tmp_path,
):
    store, record, _ = generation
    release = export_op_package(store.directory(record), tmp_path / "release")
    assert "." not in str(release)
    module = package.load_op_package(release, target_soc=record.artifact.target_soc)
    assert module.registration_path == module.vendor_path
    assert module._registration_fd is None
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == str(module.vendor_path)
