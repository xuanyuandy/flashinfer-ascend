"""T20B trust/packaging tests. All generated measurements are SYNTHETIC fixtures."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import zipfile

import pytest

from examples.two_level_gemma import configuration_registry
from flashinfer.runtime import ArtifactRef, KernelRegistry, dispatch, explain_dispatch
from flashinfer.tuning import (
    query_tuning_record, read_tuning_record, reload_tuning_records, save_tuning_record,
)
from flashinfer.tuning._release import MAX_RELEASE_BYTES, export_release_data
from tests.test_tuning import A, B, HW, SIG, STORE, fixture_record, local_db


def install_release(*records):
    """Deliberate fixture injection, bypassing release review/export identity checks."""
    root = STORE._RELEASE_DATA_ROOT
    shard = root / HW.soc_model / "measurements.sqlite3"
    for record in records:
        save_tuning_record(record, shard)
    (root / "manifest.json").write_text(json.dumps(dict(schema_version=1, socs=[HW.soc_model])))
    reload_tuning_records()
    return shard


def rewrite_payload(path, record):
    with sqlite3.connect(path) as db:
        db.execute("UPDATE measurements SET payload=?, capability_fingerprint=? WHERE record_id=?",
                   (STORE._json(record), STORE._digest(record["hardware"]), record["record_id"]))
    reload_tuning_records()


def test_valid_release_is_used_without_local_data(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    shard = install_release(record)
    before = shard.read_bytes()
    assert query_tuning_record(SIG, HW, registry=registry) == B
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == B and "tuning source=release" in decision.reason
    assert read_tuning_record(SIG, HW, registry=registry)["record_id"] == record["record_id"]
    assert shard.read_bytes() == before
    assert not local_db.parent.exists()


def test_two_argument_queries_validate_against_the_global_registry(local_db, monkeypatch):
    import flashinfer.runtime.registry as registry_module

    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    monkeypatch.setattr(registry_module, "get_global_registry", lambda: registry)
    assert query_tuning_record(SIG, HW) == B
    assert read_tuning_record(SIG, HW)["record_id"] == record["record_id"]


def test_local_winner_beats_conflicting_release_regardless_of_release_date(local_db, monkeypatch):
    registry, local = fixture_record(monkeypatch)
    _, release = fixture_record(monkeypatch, durations={A: 50, B: 100})
    release["measurement"]["finished_at"] = "2099-01-01T00:00:00+00:00"
    install_release(release)
    assert dispatch(SIG, HW, registry=registry).kernel_id == A
    save_tuning_record(local)
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == B and "tuning source=local" in decision.reason


def test_local_inconclusive_evidence_blocks_a_release_champion(local_db, monkeypatch):
    registry, release = fixture_record(monkeypatch)
    install_release(release)
    assert dispatch(SIG, HW, registry=registry).kernel_id == B
    _, local = fixture_record(monkeypatch, durations={A: 100, B: 101})
    save_tuning_record(local)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A
    assert "依据实测记录：无显著差异" in decision.reason and "source=local" in decision.reason


def test_invalid_local_evidence_allows_valid_release_fallback(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    install_release(record)
    stale = deepcopy(record)
    stale["record_id"] = "synthetic-stale-local"
    stale["candidates"][A]["descriptor_fingerprint"] = "0" * 64
    save_tuning_record(stale)
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == B
    assert "tuning source=release" in decision.reason
    assert "source=local record=synthetic-stale-local" in decision.reason
    assert "实现指纹不匹配" in decision.reason


def test_invalid_newer_local_row_does_not_hide_older_valid_evidence(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    _, stale = fixture_record(monkeypatch, durations={A: 100, B: 101})
    stale["candidates"][A]["descriptor_fingerprint"] = "0" * 64
    save_tuning_record(stale)
    assert query_tuning_record(SIG, HW, registry=registry) == B
    assert read_tuning_record(SIG, HW, registry=registry)["record_id"] == record["record_id"]


def test_injected_descriptor_fingerprint_mismatch_is_rejected(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    record["record_id"] = "synthetic-stale-descriptor"
    record["candidates"][B]["descriptor_fingerprint"] = "0" * 64
    save_tuning_record(record)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    assert read_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert record["result"]["winner"] == B and decision.kernel_id == A
    explanation = explain_dispatch(SIG, HW, registry=registry)
    assert "source=local record=synthetic-stale-descriptor" in explanation
    assert "实现指纹不匹配" in explanation and "descriptor_fingerprint" in explanation
    print("SYNTHETIC rejection fixture; not performance evidence")
    print("stored winner=%s; selected=%s" % (record["result"]["winner"], decision.kernel_id))
    print(explanation)


@pytest.mark.parametrize("changed_id", [A, B], ids=["loser", "winner"])
@pytest.mark.parametrize("change", ["source", "config", "artifact", "removed"])
def test_every_compared_identity_must_still_match(local_db, monkeypatch, changed_id, change):
    registry, record = fixture_record(monkeypatch)
    old = registry.get(changed_id)
    if change == "source":
        changed = replace(old, source=replace(old.source, digest="0" * 64))
    elif change == "config":
        changed = replace(old, tiling_schema_version=old.tiling_schema_version + 1)
    elif change == "artifact":
        changed = replace(old, artifact=ArtifactRef(
            "file", "triton_npubin", "test/kernel.bin", "0" * 64,
            HW.soc_model, "synthetic-new-build", size_bytes=1, entry_point="test_entry",
        ))
    else:
        changed = None
    current = KernelRegistry(changed if d.kernel_id == changed_id else d
                             for d in registry.descriptors() if changed is not None or d.kernel_id != changed_id)
    save_tuning_record(record)
    assert query_tuning_record(SIG, HW, registry=registry) == B  # positive control
    assert query_tuning_record(SIG, HW, registry=current) is None
    assert read_tuning_record(SIG, HW, registry=current) is None
    explanation = explain_dispatch(SIG, HW, registry=current)
    assert ("比较候选已移除或未注册" if change == "removed" else "实现指纹不匹配") in explanation
    assert "实测 tuning 记录命中" not in explanation


@pytest.mark.parametrize("changed_id", [A, B], ids=["loser", "winner"])
def test_capability_fingerprint_is_an_independent_gate(local_db, monkeypatch, changed_id):
    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    assert query_tuning_record(SIG, HW, registry=registry) == B
    record["candidates"][changed_id]["capability_fingerprint"] = "0" * 64
    rewrite_payload(local_db, record)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A and "capability_fingerprint" in decision.reason


@pytest.mark.parametrize("field", ["descriptor_fingerprint", "capability_fingerprint"])
def test_release_checks_loser_identity_too(local_db, monkeypatch, field):
    registry, record = fixture_record(monkeypatch)
    shard = install_release(record)
    assert query_tuning_record(SIG, HW, registry=registry) == B
    record["candidates"][A][field] = "0" * 64
    rewrite_payload(shard, record)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A and field in decision.reason and "source=release" in decision.reason


@pytest.mark.parametrize("hardware", [
    replace(HW, soc_model="Ascend910B"), replace(HW, aicore_count=16),
    replace(HW, ub_size_bytes=131072), type(HW)("Ascend910B", 16, 131072),
], ids=["soc", "cores", "ub", "all"])
def test_release_hardware_mismatch_is_rejected(local_db, monkeypatch, hardware):
    registry, record = fixture_record(monkeypatch)
    record["record_id"] = "synthetic-wrong-release-hardware"
    shard = install_release(record)
    assert query_tuning_record(SIG, HW, registry=registry) == B  # prove shard is consulted
    # Deliberately misfile a different machine's record in the current SoC's
    # shard, with a consistent SQL hardware index. Directory/index cannot vouch
    # for hardware; only the full stored HardwareSignature can.
    record["hardware"] = asdict(hardware)
    rewrite_payload(shard, record)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A and record["result"]["winner"] == B
    explanation = explain_dispatch(SIG, HW, registry=registry)
    assert "source=release record=synthetic-wrong-release-hardware" in explanation
    assert "HardwareSignature 不匹配" in explanation
    for key, value in asdict(hardware).items():
        if value != getattr(HW, key):
            assert "%s saved=%s current=%s" % (key, value, getattr(HW, key)) in explanation
    print("SYNTHETIC release rejection fixture; not performance evidence")
    print("stored winner=%s; selected=%s" % (record["result"]["winner"], decision.kernel_id))
    print(explanation)


@pytest.mark.parametrize("tier", ["local", "release"])
@pytest.mark.parametrize("version", [0, 2])
def test_unsupported_record_schema_is_rejected(local_db, monkeypatch, tier, version):
    registry, record = fixture_record(monkeypatch)
    path = install_release(record) if tier == "release" else local_db
    if tier == "local":
        save_tuning_record(record)
    assert query_tuning_record(SIG, HW, registry=registry) == B
    record["schema_version"] = version
    rewrite_payload(path, record)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A
    assert "record schema_version=%d expected=1" % version in decision.reason


@pytest.mark.parametrize("tier", ["local", "release"])
def test_unsupported_database_schema_is_rejected(local_db, monkeypatch, tier):
    registry, record = fixture_record(monkeypatch)
    path = install_release(record) if tier == "release" else local_db
    if tier == "local":
        save_tuning_record(record)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=2")
    reload_tuning_records()
    assert query_tuning_record(SIG, HW, registry=registry) is None
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A and "database schema_version=2 expected=1" in decision.reason


def test_legacy_t20a_database_is_still_usable(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    with sqlite3.connect(local_db) as db:
        db.execute("PRAGMA user_version=0")
    reload_tuning_records()
    assert query_tuning_record(SIG, HW, registry=registry) == B


def test_unsupported_release_manifest_schema_is_rejected(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    install_release(record)
    (STORE._RELEASE_DATA_ROOT / "manifest.json").write_text(json.dumps(dict(schema_version=2, socs=[HW.soc_model])))
    reload_tuning_records()
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A and "manifest schema_version=2 expected=1" in decision.reason


def test_unknown_hardware_values_are_not_wildcards(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    unknown = replace(HW, ub_size_bytes=0)
    record["hardware"] = asdict(unknown)
    save_tuning_record(record)
    assert query_tuning_record(SIG, unknown, registry=registry) is None
    assert "HardwareSignature 信息不完整" in explain_dispatch(SIG, unknown, registry=registry)


def test_release_cannot_bypass_existing_jit_filter(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    install_release(record)
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    decision = dispatch(SIG, HW, registry=registry, allow_reference=True)
    assert decision.backend.value == "reference"
    assert "不在当前可用候选中" in decision.reason


def test_disabled_lookup_keeps_the_t20a_all_tuning_off_contract(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    install_release(record)
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")
    assert query_tuning_record(SIG, HW, registry=registry) is None
    assert dispatch(SIG, HW, registry=registry).kernel_id == A


def test_runtime_opens_only_current_soc_and_uses_read_only_sql(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    shard = install_release(record)
    original, opened = sqlite3.connect, []

    def read_only(path, **kwargs):
        assert kwargs["uri"] and str(path).endswith("?mode=ro")
        opened.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(STORE.sqlite3, "connect", read_only)
    assert query_tuning_record(SIG, HW, registry=registry) == B
    assert len(opened) == 1 and shard.as_uri() in opened[0]
    assert query_tuning_record(SIG, replace(HW, soc_model="Ascend910B"), registry=registry) is None
    assert len(opened) == 1


def test_cached_resolution_and_dispatch_perform_no_file_or_sql_io(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    install_release(record)
    assert dispatch(SIG, HW, registry=registry).kernel_id == B

    def forbidden(*args, **kwargs):
        raise AssertionError("cache hit must not inspect files or open SQL")

    with monkeypatch.context() as guarded:
        guarded.setattr(STORE.sqlite3, "connect", forbidden)
        guarded.setattr(Path, "is_file", forbidden)
        guarded.setattr(Path, "stat", forbidden)
        assert query_tuning_record(SIG, HW, registry=registry) == B
        assert dispatch(SIG, HW, registry=registry).cache_hit


def test_returned_records_cannot_mutate_cached_evidence(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    install_release(record)
    hit = query_tuning_record(SIG, HW, registry=registry)
    hit.record["candidates"][B]["descriptor_fingerprint"] = "0" * 64
    read = read_tuning_record(SIG, HW, registry=registry)
    read["result"]["winner"] = A
    assert dispatch(SIG, HW, registry=registry).kernel_id == B


def test_export_is_reproducible_and_keeps_latest_inconclusive_per_key(local_db, monkeypatch, tmp_path):
    registry, older = fixture_record(monkeypatch)
    _, newer = fixture_record(monkeypatch, durations={A: 100, B: 101})
    older["measurement"]["finished_at"] = "2026-01-01T00:00:00+00:00"
    newer["measurement"]["finished_at"] = "2026-01-02T00:00:00+00:00"
    first, second = tmp_path / "export1", tmp_path / "export2"
    result = export_release_data([newer, older], first, registry_factory=configuration_registry)
    assert result["records"] == 1 and result["bytes"] <= MAX_RELEASE_BYTES
    assert export_release_data([older, newer], second, registry_factory=configuration_registry) == result
    assert {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()} == {
        p.relative_to(second): p.read_bytes() for p in second.rglob("*") if p.is_file()}
    monkeypatch.setattr(STORE, "_RELEASE_DATA_ROOT", first)
    assert query_tuning_record(SIG, HW, registry=registry) is None
    assert read_tuning_record(SIG, HW, registry=registry)["record_id"] == newer["record_id"]


@pytest.mark.parametrize("invalid", ["fingerprint", "hardware", "schema", "summary", "budget"])
def test_export_rejects_invalid_inputs_without_partial_output(local_db, monkeypatch, tmp_path, invalid):
    _, record = fixture_record(monkeypatch)
    if invalid == "fingerprint":
        record["candidates"][B]["descriptor_fingerprint"] = "0" * 64
    elif invalid == "hardware":
        record["hardware"]["ub_size_bytes"] = 0
    elif invalid == "schema":
        record["schema_version"] = 2
    elif invalid == "summary":
        record["result"]["winner"] = A
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        export_release_data([record], destination, registry_factory=configuration_registry,
                            max_bytes=1 if invalid == "budget" else MAX_RELEASE_BYTES)
    assert not destination.exists()


def test_export_partitions_soc_and_preserves_core_count_variants(local_db, monkeypatch, tmp_path):
    _, record = fixture_record(monkeypatch)
    variants = [record]
    for soc, cores in ((HW.soc_model, 16), ("Ascend910B", 32)):
        other = deepcopy(record)
        other["record_id"] = "synthetic-%s-%d" % (soc, cores)
        other["hardware"] = asdict(replace(HW, soc_model=soc, aicore_count=cores))
        registry = configuration_registry(soc)
        for key, identity in other["candidates"].items():
            identity["descriptor_fingerprint"] = registry.get(key).fingerprint
            identity["capability_fingerprint"] = registry.get(key).capability.fingerprint
        variants.append(other)
    destination = tmp_path / "export"
    result = export_release_data(variants, destination, registry_factory=configuration_registry)
    assert result["records"] == 3 and result["socs"] == ["Ascend910B", "Ascend950"]
    monkeypatch.setattr(STORE, "_RELEASE_DATA_ROOT", destination)
    for variant in variants:
        hardware = type(HW)(**variant["hardware"])
        registry = configuration_registry(hardware.soc_model)
        assert query_tuning_record(SIG, hardware, registry=registry) == B
        assert read_tuning_record(SIG, hardware, registry=registry)["record_id"] == variant["record_id"]


def test_empty_export_matches_checked_in_manifest(tmp_path):
    result = export_release_data([], tmp_path / "empty")
    checked_in = Path(__file__).resolve().parents[1] / "flashinfer/tuning/data/manifest.json"
    assert result == dict(records=0, socs=[], bytes=checked_in.stat().st_size)
    assert (tmp_path / "empty/manifest.json").read_bytes() == checked_in.read_bytes()


def test_checked_in_release_shards_match_manifest_and_size_budget():
    root = Path(__file__).resolve().parents[1] / "flashinfer/tuning/data"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert sorted(p.parent.name for p in root.glob("*/measurements.sqlite3")) == sorted(manifest["socs"])
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) <= MAX_RELEASE_BYTES


def test_wheel_and_sdist_include_only_release_data_and_installed_lookup_works(local_db, monkeypatch, tmp_path):
    registry, record = fixture_record(monkeypatch)
    repository = Path(__file__).resolve().parents[1]
    source, artifacts, installed = (tmp_path / p for p in ("source", "artifacts", "installed"))
    shutil.copytree(repository / "flashinfer", source / "flashinfer",
                    ignore=shutil.ignore_patterns("__pycache__", "*.sqlite3"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository / name, source / name)
    data = source / "flashinfer/tuning/data"
    shutil.rmtree(data)
    export_release_data([record], data, registry_factory=configuration_registry)
    # A local DB accidentally sitting inside the package must not enter either
    # distribution. Only the explicit SoC release pattern is package data.
    (source / "flashinfer/tuning/local.sqlite3").write_bytes(b"must not ship")
    artifacts.mkdir()
    built = subprocess.run([sys.executable, "-c",
                            "from setuptools.build_meta import build_sdist, build_wheel; "
                            "build_sdist(%r); build_wheel(%r)" % (str(artifacts), str(artifacts))],
                           cwd=source, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    expected = "flashinfer/tuning/data/Ascend950/measurements.sqlite3"
    with zipfile.ZipFile(next(artifacts.glob("*.whl"))) as wheel:
        assert expected in wheel.namelist()
        assert "flashinfer/tuning/data/manifest.json" in wheel.namelist()
        assert not any(name.endswith("local.sqlite3") for name in wheel.namelist())
        wheel.extractall(installed)
    with tarfile.open(next(artifacts.glob("*.tar.gz"))) as sdist:
        assert any(name.endswith(expected) for name in sdist.getnames())
        assert not any(name.endswith("local.sqlite3") for name in sdist.getnames())
    script = '''
import builtins, subprocess, time
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'torch_npu', 'triton', 'acl'}, name
    return original(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError('collection/compiler probe during installed lookup')
builtins.__import__ = guarded
subprocess.Popen = forbidden
time.perf_counter_ns = forbidden
import json, flashinfer
from flashinfer.runtime import HardwareSignature, InputSignature, KernelRegistry, dispatch
from flashinfer.runtime.manifest import descriptor_from_dict
from flashinfer.tuning import query_tuning_record
registry = KernelRegistry(descriptor_from_dict(d) for d in json.loads(%r))
hardware = HardwareSignature(**json.loads(%r))
signature = InputSignature(**json.loads(%r))
assert query_tuning_record(signature, hardware, registry=registry) == %r
decision = dispatch(signature, hardware, registry=registry)
assert decision.kernel_id == %r and 'tuning source=release' in decision.reason
assert dispatch(signature, hardware, registry=registry).cache_hit
print('PASS installed wheel: release hit; no provider import, collection, or compiler probe')
''' % (json.dumps([d.to_dict() for d in registry.descriptors()]), json.dumps(asdict(HW)),
       json.dumps(SIG.to_dict()), B, B)
    import os
    environment = dict(os.environ, PYTHONPATH=str(installed))
    checked = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment,
                             capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "PASS installed wheel" in checked.stdout
    print("PASS wheel/sdist include SoC release DB and manifest; exclude local DB")
    print(checked.stdout.strip())
