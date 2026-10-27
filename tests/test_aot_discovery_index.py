"""Metadata-index/error contracts; real execution lives in the real_compile suite."""

from dataclasses import replace
from pathlib import Path
import platform
import shutil

import pytest

from flashinfer.aot.build import export_op_package
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
from tests.test_ascendc_op_package import elf_fixture, generation, unit, write_fixture  # noqa: F401


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
    assert first.roots == (Path(discovery.__file__).resolve().parents[1] / "data/op_packages",)
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
