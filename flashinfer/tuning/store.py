"""Append-only local SQLite measurements, read-only lookup, explicit reload."""

from dataclasses import asdict
from datetime import datetime
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Optional

from ..runtime.hardware import HardwareSignature
from ..runtime.schema import ShapeRange
from ..runtime.signature import InputSignature


_revision = 0


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _db_path():
    value = os.environ.get("FLASHINFER_NPU_TUNING_DB")
    if value is None:
        value = str(Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
                    / "flashinfer" / "tuning.sqlite3")
    return os.path.abspath(os.path.expanduser(value)) if value else ""


def tuning_cache_token():
    # No stat, SQL, device probe, measurement, or file creation on cache hits.
    return (_db_path(), _revision)


def reload_tuning_records():
    """Refresh local reads/decisions (also called after our writes).

    Another process's writes require this call or a process restart in T20A.
    Release DB priorities and automatic invalidation belong to T20B.
    """
    global _revision
    _revision += 1
    _lookup.cache_clear()
    from ..runtime.dispatch import clear_dispatch_cache
    clear_dispatch_cache()


def _matches(bucket, signature):
    actual = signature.to_dict()
    actual.pop("op")
    expected = bucket["signature"]
    shapes = actual.pop("shapes")
    template = expected["shapes"]
    if actual != {k: v for k, v in expected.items() if k != "shapes"}:
        return False
    if len(shapes) != len(template) or any(len(a) != len(b) for a, b in zip(shapes, template)):
        return False
    if any(e is not None and e != a for shape, pattern in zip(shapes, template)
           for a, e in zip(shape, pattern)):
        return False
    covered = set()
    for raw in bucket["shape_ranges"]:
        bound = ShapeRange.from_dict(raw)
        shape = shapes[bound.tensor_index]
        if not bound.minimum <= shape[bound.axis] <= bound.maximum:
            return False
        covered.add((bound.tensor_index, bound.axis % len(shape)))
    return all((i, j) in covered for i, shape in enumerate(template)
               for j, value in enumerate(shape) if value is None)


def _validate(record):
    from .benchmark import BenchmarkConfig, _analyze
    if record["schema_version"] != 1 or record["op"] != record["signature"]["op"]:
        raise ValueError("unsupported tuning record")
    measurement = record["measurement"]
    for key in ("started_at", "finished_at"):
        if datetime.fromisoformat(measurement[key]).tzinfo is None:
            raise ValueError("measurement timestamp must include timezone")
    if (not record["record_id"] or not isinstance(measurement["timer"], str)
            or measurement["unit"] != "us/invocation"):
        raise ValueError("measurement identity/timer/unit missing")
    for identity in record["candidates"].values():
        if any(not re.fullmatch(r"[0-9a-f]{64}", identity[key])
               for key in ("descriptor_fingerprint", "capability_fingerprint")):
            raise ValueError("invalid candidate identity")
    for snapshot in (measurement["machine_before"], measurement["machine_after"]):
        load = snapshot["load_average"]
        if len(load) != 3 or any(not 0 <= value < float("inf") for value in load):
            raise ValueError("invalid machine load observation")
    activity = measurement["machine_activity"]
    if any(value is not None and not 0 <= value <= 100 for value in (
            activity["system_cpu_busy_percent"], activity["affinity_cpu_busy_percent"])):
        raise ValueError("invalid machine CPU observation")
    config = BenchmarkConfig(**record["measurement"]["config"])
    samples = record["measurement"]["samples_us"]
    orders = record["measurement"]["orders"]
    if (len(samples) < 2 or set(samples) != set(record["candidates"])
            or len(orders) != config.rounds
            or any(len(o) != len(samples) or set(o) != set(samples) for o in orders)
            or any(len(s) != config.rounds or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float("inf")
                for v in s) for s in samples.values())):
        raise ValueError("invalid tuning samples")
    if record["result"] != _analyze(samples, config, orders):
        raise ValueError("result does not match raw measurements")
    if not _matches(record["bucket"], InputSignature(**record["signature"])):
        raise ValueError("measurement is outside its bucket")


def save_tuning_record(record, path=None):
    """Persist a complete measurement, including inconclusive results; never at import."""
    _validate(record)
    target = Path(path if path is not None else _db_path()).expanduser()
    if path is None and not _db_path():
        raise ValueError("FLASHINFER_NPU_TUNING_DB is disabled")
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(target), timeout=5)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS measurements ("
                   "sequence INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT UNIQUE NOT NULL, "
                   "op TEXT NOT NULL, bucket_key TEXT NOT NULL, capability_fingerprint TEXT NOT NULL, "
                   "collected_at TEXT NOT NULL, payload TEXT NOT NULL)")
        db.execute("CREATE INDEX IF NOT EXISTS tuning_lookup ON measurements "
                   "(op, capability_fingerprint, sequence DESC)")
        db.execute("INSERT INTO measurements (record_id, op, bucket_key, capability_fingerprint, "
                   "collected_at, payload) VALUES (?, ?, ?, ?, ?, ?)",
                   (record["record_id"], record["op"], _digest(record["bucket"]),
                    _digest(record["hardware"]), record["measurement"]["finished_at"], _json(record)))
        db.commit()
    reload_tuning_records()
    return record["record_id"]


@lru_cache(maxsize=512)
def _lookup(path, revision, signature, hardware):
    if not path:
        return None
    try:
        if not Path(path).is_file():
            return None
        with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=.1)) as db:
            rows = db.execute("SELECT payload FROM measurements WHERE op=? AND "
                              "capability_fingerprint=? ORDER BY sequence DESC",
                              (signature.op, _digest(asdict(hardware))))
            for (payload,) in rows:
                record = json.loads(payload)
                if (record["hardware"] != asdict(hardware) or record["op"] != signature.op
                        or not _matches(record["bucket"], signature)):
                    continue
                _validate(record)
                # A newer inconclusive measurement suppresses an older winner.
                return record
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, IndexError):
        return None
    return None


def read_tuning_record(signature, hardware):
    """Read matching evidence, including no-significant-difference, without benchmarking."""
    record = _lookup(*tuning_cache_token(), signature, hardware)
    return json.loads(_json(record)) if record is not None else None


class _TuningHit(str):
    """Preserve Optional[str] lookup ABI while carrying evidence to dispatch."""

    def __new__(cls, record):
        hit = str.__new__(cls, record["result"]["winner"])
        hit.record = record
        return hit


def query_tuning_record(signature: InputSignature, hardware: HardwareSignature) -> Optional[str]:
    """Return a measured champion ID, or None for missing/inconclusive local data."""
    record = _lookup(*tuning_cache_token(), signature, hardware)
    if record is None or record["result"]["winner"] is None:
        return None
    return _TuningHit(record)


def _evidence_description(record):
    measurement, result = record["measurement"], record["result"]
    config = measurement["config"]
    summaries = "; ".join("%s median=%.6f us MAD=%.6f us IQR=%.6f us" % (
        key, values["median_us"], values["mad_us"], values["iqr_us"])
        for key, values in sorted(result["distributions"].items()))
    activity = measurement["machine_activity"]
    comparisons = "; ".join("gap=%.6f us required>%.6f us paired_wins=%d/%d p=%.6g" % (
        c["median_gap_us"], c["required_gap_us"], c["paired_practical_wins"],
        c["paired_rounds"], c["sign_test_p"]) for c in result["comparisons"])
    return ("依据实测记录：%s；record=%s；%s；warmup=%d rounds=%d iterations=%d；"
            "可信度判据=%s family_alpha=%g pair_alpha=%g min_gap=%.1f%%；采集=%s；load1=%.2f；affinity_cpu_busy=%s%%；%s；%s" % (
                "显著冠军" if result["winner"] else "无显著差异",
                record["record_id"], measurement["timer"], config["warmup"], config["rounds"],
                config["iterations"], result["rule"], config["family_alpha"],
                config["family_alpha"] / (len(record["candidates"]) * (len(record["candidates"]) - 1)),
                100 * config["minimum_relative_gap"], measurement["finished_at"],
                measurement["machine_before"]["load_average"][0],
                activity["affinity_cpu_busy_percent"], summaries, comparisons))
