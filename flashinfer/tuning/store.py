"""Local/release evidence, shared validity checks, and read-only runtime lookup."""

from dataclasses import asdict, dataclass, replace
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
_SCHEMA_VERSION = 1
_RELEASE_DATA_ROOT = Path(__file__).with_name("data")


@dataclass(frozen=True)
class _TuningResolution:
    record: Optional[dict] = None
    source: str = ""
    rejections: tuple = ()


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
    return (_db_path(), str(_RELEASE_DATA_ROOT), _revision)


def reload_tuning_records():
    """Refresh both tiers and decisions (also called after our local writes).

    Another process's writes or an in-place package replacement require this
    call or a process restart. Validity is checked on every fresh resolution.
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
    if (type(record["schema_version"]) is not int
            or record["schema_version"] != _SCHEMA_VERSION
            or record["op"] != record["signature"]["op"]):
        raise ValueError("unsupported tuning record")
    hardware = record["hardware"]
    if (set(hardware) != {"soc_model", "aicore_count", "ub_size_bytes"}
            or not isinstance(hardware["soc_model"], str)
            or any(type(hardware[k]) is not int or hardware[k] < 0
                   for k in ("aicore_count", "ub_size_bytes"))):
        raise ValueError("invalid HardwareSignature")
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
    _append_record(record, target)
    reload_tuning_records()
    return record["record_id"]


def _append_record(record, target):
    """Shared writer for validated local records and the offline release exporter."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(target), timeout=5)) as db:
        if db.execute("PRAGMA user_version").fetchone()[0] not in (0, _SCHEMA_VERSION):
            raise ValueError("unsupported tuning database schema_version")
        db.execute("PRAGMA user_version=%d" % _SCHEMA_VERSION)
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


def _known_hardware(hardware):
    return (bool(hardware.soc_model) and hardware.soc_model != "unknown"
            and hardware.aicore_count > 0 and hardware.ub_size_bytes > 0)


def _identity_reasons(record, hardware, identities):
    """Reuse T20A's identities; a comparison is stale if ANY participant changed."""
    reasons = []
    actual = asdict(hardware)
    differences = ["%s saved=%s current=%s" % (key, record["hardware"].get(key), value)
                   for key, value in actual.items() if record["hardware"].get(key) != value]
    if differences or record["hardware"] != actual:
        reasons.append("HardwareSignature 不匹配：" + ", ".join(differences))
    elif not _known_hardware(hardware):
        reasons.append("HardwareSignature 信息不完整（SoC、核数、UB 必须已知）")
    current = {key: (descriptor, capability) for key, descriptor, capability in identities}
    for key, saved in sorted(record["candidates"].items()):
        if key not in current:
            reasons.append("比较候选已移除或未注册：%s" % key)
            continue
        for name, value in zip(("descriptor_fingerprint", "capability_fingerprint"), current[key]):
            if saved[name] != value:
                reasons.append("实现指纹不匹配：%s %s saved=%s current=%s" % (
                    key, name, saved[name][:12], value[:12]))
    return reasons


def _reject(rejections, source, record_id, reason):
    # Bound explain output even if a long local history contains stale rows.
    if len(rejections) < 8:
        rejections.append("tuning 记录拒绝：source=%s record=%s；%s" % (
            source, record_id, reason))


def _read_database(path, source, signature, hardware, identities, rejections):
    try:
        if not Path(path).is_file():
            return None
        with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=.1)) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            # T20A databases used SQLite's default user_version=0; their payloads
            # must still pass the exact record schema check below.
            if version not in (0, _SCHEMA_VERSION):
                _reject(rejections, source, "-", "database schema_version=%s expected=%s" % (
                    version, _SCHEMA_VERSION))
                return None
            # Read other hardware variants too, so a bucket hit with incompatible
            # hardware can be explained. Never trust a shard name or SQL index
            # as evidence of hardware compatibility.
            rows = db.execute("SELECT record_id, capability_fingerprint, payload FROM measurements "
                              "WHERE op=? ORDER BY sequence DESC", (signature.op,))
            for record_id, hardware_digest, payload in rows:
                try:
                    record = json.loads(payload)
                    version = record["schema_version"]
                    if type(version) is not int or version != _SCHEMA_VERSION:
                        _reject(rejections, source, record_id, "record schema_version=%s expected=%s" % (
                            version, _SCHEMA_VERSION))
                        continue
                    if record["op"] != signature.op or not _matches(record["bucket"], signature):
                        continue
                    reasons = _identity_reasons(record, hardware, identities)
                    if hardware_digest != _digest(record["hardware"]):
                        reasons.append("HardwareSignature 索引与 payload 不一致")
                    if reasons:
                        _reject(rejections, source, record_id, "；".join(reasons))
                        continue
                    _validate(record)
                    # Latest VALID evidence wins, including an inconclusive
                    # measurement. Invalid rows cannot hide older valid evidence.
                    return record
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                    _reject(rejections, source, record_id, "记录损坏：%s" % error)
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, IndexError):
        _reject(rejections, source, "-", "数据库不可读或格式损坏")
    return None


def _release_database(root, hardware, rejections):
    try:
        manifest_path = Path(root) / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        if (type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != _SCHEMA_VERSION):
            _reject(rejections, "release", "-", "manifest schema_version=%s expected=%s" % (
                manifest["schema_version"], _SCHEMA_VERSION))
            return None
        socs = manifest["socs"]
        if (not isinstance(socs, list)
                or any(not isinstance(soc, str) or not re.fullmatch(r"[A-Za-z0-9_]+", soc) for soc in socs)):
            raise ValueError("invalid SoC manifest")
        if hardware.soc_model in socs:
            return Path(root) / hardware.soc_model / "measurements.sqlite3"
    except (OSError, ValueError, KeyError, TypeError):
        _reject(rejections, "release", "-", "发版清单不可读或格式损坏")
    return None


@lru_cache(maxsize=512)
def _lookup(path, release_root, revision, signature, hardware, identities):
    if not path:
        return _TuningResolution()
    rejections = []
    record = _read_database(path, "local", signature, hardware, identities, rejections)
    if record is not None:
        return _TuningResolution(record, "local", tuple(rejections))
    release_path = _release_database(release_root, hardware, rejections)
    if release_path is not None:
        record = _read_database(release_path, "release", signature, hardware, identities, rejections)
        if record is not None:
            return _TuningResolution(record, "release", tuple(rejections))
    return _TuningResolution(rejections=tuple(rejections))


def _resolve_tuning_records(signature, hardware, registry=None):
    if not _db_path():
        return _TuningResolution()
    if registry is None:
        from ..runtime.registry import get_global_registry
        registry = get_global_registry()
    identities = tuple((d.kernel_id, d.fingerprint, d.capability.fingerprint)
                       for d in registry.descriptors(op=signature.op))
    return _lookup(*tuning_cache_token(), signature, hardware, identities)


def read_tuning_record(signature, hardware, *, registry=None):
    """Return a copy of valid local/release evidence (also inconclusive results).

    Two-argument calls use the global registry. Supply a private registry when
    querying measurements collected from explicitly supplied private kernels.
    """
    record = _resolve_tuning_records(signature, hardware, registry).record
    return json.loads(_json(record)) if record is not None else None


class _TuningHit(str):
    """Preserve Optional[str] lookup ABI while carrying evidence to dispatch."""

    def __new__(cls, resolution):
        record = json.loads(_json(resolution.record))
        hit = str.__new__(cls, record["result"]["winner"])
        hit.record = record
        hit.resolution = replace(resolution, record=record)
        return hit


def query_tuning_record(signature: InputSignature, hardware: HardwareSignature, *, registry=None) -> Optional[str]:
    """Return a valid local/release champion ID; missing/stale/inconclusive -> None."""
    resolution = _resolve_tuning_records(signature, hardware, registry)
    record = resolution.record
    if record is None or record["result"]["winner"] is None:
        return None
    return _TuningHit(resolution)


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
