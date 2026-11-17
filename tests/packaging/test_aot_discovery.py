"""aot discovery regression coverage."""


from dataclasses import replace
from pathlib import Path
import platform
import shutil

import pytest

from flashinfer.jit._legacy_recipe import export_op_package
from flashinfer.jit import discovery
from flashinfer.jit import DiskJitBuildStore
from flashinfer.jit.build import _verify_output
from flashinfer.jit.discovery import (
    AotCorruptError, AotLoadError, AotMissingError, AotPackageIndex,
    get_op_package_index, invalidate_op_package_index,
)
from flashinfer.runtime import Backend, CompileMode, KernelDescriptor, KernelLaunchABI

# Reuse the existing verifier's synthetic ELF/store fixture; these bytes are
# never executed. End-to-end tests separately compile and execute real packages.
from tests.packaging.test_ascendc_op_package import elf_fixture, generation, unit, write_fixture


@pytest.fixture
def shipped(generation, unit, tmp_path):
    recipe, _, spec = unit
    descriptor = KernelDescriptor(recipe.kernel_id, "test.t13.matmul_probe", Backend.ASCENDC,
        compile_mode=CompileMode.AOT, source=recipe.source, target_soc=recipe.target_soc,
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
                                  ("workspace", "workspace_bytes", "executor", "stream")))
    store, record, paths = generation
    root = tmp_path / "release"
    directory = export_op_package(store.directory(record), root)
    return root, directory, spec, descriptor, paths


def find(index, shipped, **kwargs):
    _, _, spec, descriptor, _ = shipped
    return index.find(spec, descriptor=kwargs.get("descriptor", descriptor),
                      target_soc=kwargs.get("target_soc", spec.target_soc))


def test_default_wheel_path_and_environment_indexes(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASHINFER_AOT_PATHS", raising=False)
    invalidate_op_package_index()
    first = get_op_package_index()
    assert first.roots == (Path(discovery.__file__).resolve().parents[2] / "build/generated/flashinfer/data/op_packages",)
    assert get_op_package_index() is first
    monkeypatch.setenv("FLASHINFER_AOT_PATHS", str(tmp_path))
    second = get_op_package_index()
    assert second is not first and second.roots[-1] == tmp_path
    invalidate_op_package_index()
    assert get_op_package_index() is not second


def test_cached_index_does_not_rescan_root(shipped, monkeypatch):
    index = AotPackageIndex((shipped[0],))
    assert find(index, shipped).directory == shipped[1]

    def no_scan():
        pytest.fail("unchanged roots must not be rescanned")

    monkeypatch.setattr(index, "_directories", no_scan)
    assert find(index, shipped).spec == shipped[2]


def test_addition_and_removal_invalidate_missing_and_hit(shipped, tmp_path):
    root, directory, _, _, _ = shipped
    spare = tmp_path / "saved"
    shutil.move(directory, spare)
    index = AotPackageIndex((root,))
    with pytest.raises(AotMissingError, match="no matching.*found 0 candidates"):
        find(index, shipped)
    shutil.move(spare, directory)
    assert find(index, shipped).directory == directory
    shutil.move(directory, spare)
    with pytest.raises(AotMissingError, match="no matching.*found 0 candidates"):
        find(index, shipped)


def test_manifest_edit_invalidates_cached_metadata(shipped):
    index = AotPackageIndex((shipped[0],))
    find(index, shipped)
    (shipped[1] / "manifest.json").write_text("{broken JSON")
    with pytest.raises(AotCorruptError, match="corrupt.*JSONDecodeError"):
        find(index, shipped)


def test_cached_match_never_hides_payload_corruption(shipped):
    index = AotPackageIndex((shipped[0],))
    match = find(index, shipped)
    (shipped[1] / shipped[4]["binary"]).write_bytes(b"digest corruption")
    with pytest.raises(AotCorruptError, match="inventory.*mismatch"):
        find(index, shipped)
    with pytest.raises(AotCorruptError, match="inventory.*mismatch"):
        match.load()


@pytest.mark.parametrize("direct_release", (False, True))
def test_deleted_manifest_is_corrupt_not_missing(shipped, direct_release):
    (shipped[1] / "manifest.json").unlink()
    with pytest.raises(AotCorruptError, match="manifest.json"):
        find(AotPackageIndex((shipped[1] if direct_release else shipped[0],)), shipped)


@pytest.mark.parametrize("kind", ("soc", "spec", "abi"))
def test_incompatible_reasons_are_retained(shipped, kind):
    index = AotPackageIndex((shipped[0],))
    spec, descriptor, target = shipped[2], shipped[3], shipped[2].target_soc
    if kind == "soc":
        target = "ascend910b1"
    elif kind == "spec":
        spec = replace(spec, specialization_fingerprint="b" * 64)
    else:
        descriptor = replace(descriptor, launch_abi=replace(descriptor.launch_abi,
                                                            metadata_schema_version=2))
    with pytest.raises(AotMissingError) as caught:
        index.find(spec, descriptor=descriptor, target_soc=target)
    assert caught.value.candidate_count == 1
    assert caught.value.rejections[0].category == "incompatible"
    expected = {"soc": "target_soc mismatch", "spec": "spec fingerprint mismatch",
                "abi": "descriptor/ABI fingerprint mismatch"}[kind]
    assert expected in str(caught.value)
    assert "found 1 candidate(s), all incompatible" in str(caught.value)


def test_incompatible_candidate_does_not_hide_later_match(shipped, unit, tmp_path):
    recipe = unit[0]
    other_spec = replace(shipped[2], specialization_fingerprint="b" * 64)
    store = DiskJitBuildStore(tmp_path / "other-store")
    with store.lock(other_spec), store.staging(other_spec) as staging:
        write_fixture(staging, recipe, other_spec)
        record = store.publish(other_spec, staging, _verify_output(other_spec, staging))
    incompatible = export_op_package(store.directory(record), tmp_path / "incompatible")
    index = AotPackageIndex((incompatible, shipped[0]))
    assert find(index, shipped).directory == shipped[1]


@pytest.mark.parametrize("error", (OSError("dlopen failure"), AttributeError("missing symbol"),
    RuntimeError("CANN op-package registration is process-scoped; load another package in a fresh process")))
def test_load_errors_preserve_original_exception_and_never_try_other_candidates(shipped, monkeypatch, error):
    match = find(AotPackageIndex((shipped[0],)), shipped)
    calls = []

    def failed_load(directory, *, target_soc):
        calls.append((directory, target_soc))
        raise error

    monkeypatch.setattr(discovery, "load_op_package", failed_load)
    with pytest.raises(AotLoadError) as caught:
        match.load()
    assert caught.value.__cause__ is error
    assert str(error) in str(caught.value)
    assert len(calls) == 1


def test_explicit_invalidation_refreshes_retained_index(shipped):
    index = AotPackageIndex((shipped[0],))
    find(index, shipped)
    assert index._entries
    index.invalidate()
    assert not index._entries
    assert find(index, shipped).directory == shipped[1]


def test_verified_manifest_with_missing_aclnn_symbol_is_load_error(shipped, unit, tmp_path):
    recipe, _, spec = unit
    original = find(AotPackageIndex((shipped[0],)), shipped)
    store = DiskJitBuildStore(tmp_path / "missing-symbol-store")
    with store.lock(spec), store.staging(spec) as staging:
        paths = write_fixture(staging, recipe, spec)
        (staging / paths["api"]).write_bytes(elf_fixture((recipe.host_entry,),
            machine={"x86_64": 62, "aarch64": 183}[platform.machine()], kind=3))
        # Inventory/digests are valid. The missing host entry is a load contract
        # failure, distinct from corruption of those recorded bytes.
        record = store.publish(spec, staging, original.record.artifact)
    directory = store.directory(record)
    with pytest.raises(AotLoadError, match="no ACLNN symbol.*GetWorkspaceSize") as caught:
        find(AotPackageIndex((directory,)), shipped)
    assert isinstance(caught.value.__cause__, discovery.OpPackageSymbolError)


import ctypes as C
import errno
import os
from types import SimpleNamespace


from flashinfer.jit import op_package as package
from tests.packaging.test_ascendc_op_package import generation, unit


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
        if active is not None:
            for _, module in active.values():
                if module._registration_fd is not None:
                    os.close(module._registration_fd)


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


import json


from flashinfer.jit.cache import verify_build_directory
from flashinfer.runtime import SchemaError


@pytest.mark.parametrize("damage", ["digest", "size", "missing", "extra"])
def test_release_inventory_corruption_names_file_and_check(generation, damage):
    store, record, paths = generation
    directory = store.directory(record)
    relative = paths["tiling"]
    path = directory / relative
    if damage == "digest":
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        check = "digest mismatch"
    elif damage == "size":
        path.write_bytes(path.read_bytes() + b"changed")
        check = "size mismatch"
    elif damage == "missing":
        path.unlink()
        check = "missing file"
    else:
        relative = "packages/unexpected.bin"
        (directory / relative).write_bytes(b"not published")
        check = "unexpected file"
    with pytest.raises(SchemaError, match="inventory mismatch") as failed:
        verify_build_directory(directory)
    assert relative in str(failed.value)
    assert check in str(failed.value)


def test_release_manifest_digest_names_manifest_and_metadata(generation):
    store, record, paths = generation
    directory = store.directory(record)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][paths["tiling"]]["digest"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(SchemaError, match="manifest digest mismatch") as failed:
        verify_build_directory(directory)
    assert "manifest.json metadata (spec/files)" in str(failed.value)
