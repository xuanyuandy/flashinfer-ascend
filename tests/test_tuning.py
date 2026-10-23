"""Synthetic HOST fixtures test decisions only; never performance evidence."""

from dataclasses import replace
import importlib
import json
import sqlite3
import subprocess
import sys

import pytest

from examples.two_level_gemma import OP, configuration_registry
from flashinfer.runtime import HardwareSignature, InputSignature, KernelRegistry, dispatch, explain_dispatch
from flashinfer.tuning import (
    BenchmarkCandidate, BenchmarkConfig, collect_tuning_record, query_tuning_record,
    read_tuning_record, reload_tuning_records, save_tuning_record,
)
from flashinfer.tuning.benchmark import _analyze


BENCH = importlib.import_module("flashinfer.tuning.benchmark")
A, B = (f"triton/t19a/gemma/block{block}" for block in (128, 256))
SIG = InputSignature(OP, dtypes=("float32",), shapes=((4, 128),))
HW = HardwareSignature("Ascend950", 32, 262144)


def orders(keys, n=31):
    return [keys[i % len(keys):] + keys[:i % len(keys)] for i in range(n)]


def analyze(a, b):
    return _analyze({A: a, B: b}, BenchmarkConfig(), orders([A, B]))


@pytest.mark.parametrize("kwargs", [dict(warmup=0), dict(rounds=24), dict(iterations=0),
                                    dict(rounds=31.0), dict(noise_multiplier=2),
                                    dict(family_alpha=.1), dict(minimum_relative_gap=0)])
def test_reject_weak_or_invalid_collection_settings(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_clear_gap_accepts_a_champion():
    result = analyze([50 + i % 3 for i in range(31)], [100 + i % 3 for i in range(31)])
    assert result["winner"] == A
    assert result["comparisons"][0]["sign_test_p"] == 2 ** -31


def test_small_gap_is_inconclusive_despite_identical_rounds():
    assert analyze([100] * 31, [101] * 31)["winner"] is None


def test_spread_blocks_apparent_median_win():
    result = analyze([80 + (i % 5) * 10 for i in range(31)], [110 + (i % 5) * 10 for i in range(31)])
    assert result["comparisons"][0]["median_gap_us"] > 3
    assert result["comparisons"][0]["required_gap_us"] > 30
    assert result["winner"] is None


def test_paired_sign_test_rejects_an_unstable_advantage_even_with_zero_mad():
    result = analyze([50] * 16 + [110] * 15, [100] * 31)
    assert result["distributions"][A]["mad_us"] == 0
    assert result["comparisons"][0]["sign_test_p"] > .01
    assert result["winner"] is None


def test_execution_order_reversal_prevents_champion():
    result = analyze([50 if i % 2 else 110 for i in range(31)], [100] * 31)
    assert min(result["comparisons"][0]["order_gaps_us"]) < 0
    assert result["winner"] is None


def test_must_beat_every_candidate_and_correct_for_multiple_comparisons():
    samples = {"a": [50] * 31, "b": [100] * 31, "c": [51] * 31}
    result = _analyze(samples, BenchmarkConfig(), orders(list(samples)))
    assert len(result["comparisons"]) == 2
    assert result["winner"] is None


@pytest.fixture
def local_db(monkeypatch, tmp_path):
    path = tmp_path / "runtime" / "tuning.sqlite3"
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", str(path))
    reload_tuning_records()
    yield path
    reload_tuning_records()


def fixture_record(monkeypatch, *, durations=None):
    """Inject a deterministic clock, conspicuously marked as a synthetic fixture."""
    registry = configuration_registry(HW.soc_model)
    durations = durations or {A: 100, B: 50}
    current, clock_calls, preparations = [A], [0], []

    def clock():
        clock_calls[0] += 1
        return 0 if clock_calls[0] % 2 else int(durations[current[0]] * 1000)

    def candidate(key):
        def prepare(n):
            current[0] = key
            preparations.append((key, n))
            return lambda: None
        return BenchmarkCandidate(registry.get(key), prepare)

    monkeypatch.setattr(BENCH.time, "perf_counter_ns", clock)
    monkeypatch.setattr(BENCH.time, "sleep", lambda _: None)
    record = collect_tuning_record(
        SIG, HW, [candidate(B), candidate(A)], synchronize=lambda: None,
        config=BenchmarkConfig(warmup=2, rounds=25, iterations=1),
        metadata={"fixture": "SYNTHETIC UNIT TEST ONLY; NOT PERFORMANCE EVIDENCE"},
    )
    assert len(preparations) == 2 * (2 + 25)
    assert clock_calls[0] == 2 * 2 * 25
    assert record["measurement"]["orders"] == orders([A, B], 25)
    return registry, record


def test_actual_store_hit_overrides_heuristic_and_explains_confidence(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    assert dispatch(SIG, HW, registry=registry).kernel_id == A
    assert dispatch(SIG, HW, registry=registry).cache_hit
    save_tuning_record(record)
    assert query_tuning_record(SIG, HW) == B
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == B and not decision.cache_hit
    assert decision.stack_id == "triton" and decision.fallback_chain == (A,)
    explanation = explain_dispatch(SIG, HW, registry=registry)
    for text in ("依据实测记录：显著冠军", "MAD=", "IQR=", "rounds=25", "alpha=0.01", "采集=", "load1="):
        assert text in explanation
    with sqlite3.connect(local_db) as db:
        assert db.execute("SELECT count(*) FROM measurements").fetchone() == (1,)
    assert read_tuning_record(SIG, HW) == json.loads(json.dumps(record))


def test_inconclusive_measurement_suppresses_older_winner(local_db, monkeypatch):
    registry, first = fixture_record(monkeypatch)
    save_tuning_record(first)
    assert dispatch(SIG, HW, registry=registry).kernel_id == B
    _, second = fixture_record(monkeypatch, durations={A: 100, B: 101})
    save_tuning_record(second)
    assert query_tuning_record(SIG, HW) is None
    assert read_tuning_record(SIG, HW)["result"]["status"] == "no_significant_difference"
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A
    assert "确定性启发式" in decision.reason and "依据实测记录：无显著差异" in decision.reason


def test_missing_or_disabled_database_never_creates_files(local_db, monkeypatch):
    assert query_tuning_record(SIG, HW) is None
    assert read_tuning_record(SIG, HW) is None
    assert not local_db.parent.exists()
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")
    assert query_tuning_record(SIG, HW) is None


def test_corrupt_database_preserves_deterministic_fallback(local_db):
    local_db.parent.mkdir()
    local_db.write_bytes(b"corrupt SQLite database")
    registry = configuration_registry(HW.soc_model)
    decision = dispatch(SIG, HW, registry=registry)
    assert decision.kernel_id == A and decision.reason.startswith("确定性启发式")


def test_path_switch_does_not_reuse_old_dispatch_decision(local_db, monkeypatch, tmp_path):
    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    assert dispatch(SIG, HW, registry=registry).kernel_id == B
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", str(tmp_path / "other.sqlite3"))
    assert dispatch(SIG, HW, registry=registry).kernel_id == A


@pytest.mark.parametrize("signature,hardware", [
    (replace(SIG, op="other"), HW),
    (replace(SIG, dtypes=("float16",)), HW),
    (replace(SIG, shapes=((5, 128),)), HW),
    (replace(SIG, shapes=((4, 129),)), HW),
    (replace(SIG, shapes=((4, 128, 1),)), HW),
    (replace(SIG, layouts=("NHD",)), HW),
    (replace(SIG, physical_layouts=("NZ",)), HW),
    (replace(SIG, specializations=(("eps", "1e-5"),)), HW),
    (SIG, replace(HW, aicore_count=16)),
    (SIG, replace(HW, ub_size_bytes=0)),
    (SIG, replace(HW, soc_model="Ascend910B")),
])
def test_bucket_semantics_and_hardware_are_isolated(local_db, monkeypatch, signature, hardware):
    _, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    assert query_tuning_record(signature, hardware) is None


def test_bucket_reuses_t63_ranges_and_keeps_unranged_axes_exact(local_db, monkeypatch):
    _, record = fixture_record(monkeypatch)
    assert record["bucket"]["signature"]["shapes"] == [[4, None]]
    assert [(r["minimum"], r["maximum"], r["axis"]) for r in record["bucket"]["shape_ranges"]] == [(1, 128, 1), (1, 256, 1)]
    save_tuning_record(record)
    assert query_tuning_record(replace(SIG, shapes=((4, 64),)), HW) == B
    assert query_tuning_record(replace(SIG, shapes=((4, 256),)), HW) is None


def test_changed_winner_identity_cannot_use_old_measurement(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    changed = KernelRegistry(replace(d, priority=-1) if d.kernel_id == B else d
                             for d in registry.descriptors())
    decision = dispatch(SIG, HW, registry=changed)
    assert decision.kernel_id == A and "实现指纹不匹配" in decision.reason


def test_filtering_and_explicit_overrides_still_take_precedence(local_db, monkeypatch):
    registry, record = fixture_record(monkeypatch)
    save_tuning_record(record)
    assert dispatch(SIG, HW, registry=registry, kernel_id=A).kernel_id == A
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    assert dispatch(SIG, HW, registry=registry, allow_reference=True).backend.value == "reference"


def test_tampered_summary_is_rejected_before_write(local_db, monkeypatch):
    _, record = fixture_record(monkeypatch)
    record["result"]["winner"] = A
    with pytest.raises(ValueError, match="raw measurements"):
        save_tuning_record(record)
    assert not local_db.exists()


def test_ineligible_candidate_is_rejected_without_execution(monkeypatch):
    descriptor = configuration_registry(HW.soc_model).get(A)
    def forbidden(*args):
        raise AssertionError("measurement must not run")
    with pytest.raises(ValueError, match="ineligible"):
        collect_tuning_record(replace(SIG, shapes=((4, 256),)), HW,
                              [BenchmarkCandidate(descriptor, forbidden),
                               BenchmarkCandidate(replace(descriptor, kernel_id="triton/other"), forbidden)],
                              synchronize=forbidden)


def test_import_and_dispatch_cannot_collect_compile_or_import_providers(local_db):
    script = '''
import builtins, subprocess, time
original = builtins.__import__
blocked = []
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'torch', 'torch_npu', 'triton', 'acl'}:
        blocked.append(name)
        raise AssertionError('provider import: ' + name)
    return original(name, globals, locals, fromlist, level)
def forbidden(*args, **kwargs):
    raise AssertionError('measurement or compilation on import/dispatch')
builtins.__import__ = guarded
subprocess.Popen = forbidden
time.perf_counter_ns = forbidden
from examples.two_level_gemma import OP, configuration_registry
from flashinfer.runtime import HardwareSignature, InputSignature, dispatch
from flashinfer.tuning import query_tuning_record
hw = HardwareSignature('Ascend950', 32)
sig = InputSignature(OP, dtypes=('float32',), shapes=((4,128),))
assert query_tuning_record(sig, hw) is None
reg = configuration_registry(hw.soc_model)
assert dispatch(sig, hw, registry=reg).kernel_id.endswith('block128')
assert dispatch(sig, hw, registry=reg).cache_hit
assert not blocked
print('PASS import/dispatch: no providers, timing, compilation, or DB creation')
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASS import/dispatch" in result.stdout
    assert not local_db.parent.exists()
