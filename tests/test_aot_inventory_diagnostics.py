"""Shared release verification identifies the bytes that failed validation."""

import json

import pytest

from flashinfer.jit.cache import verify_build_directory
from flashinfer.runtime import SchemaError
from tests.test_ascendc_op_package import generation, unit  # noqa: F401


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
