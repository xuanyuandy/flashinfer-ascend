"""Explicit, reproducible release export; never imported by runtime lookup."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile

from ..runtime.hardware import HardwareSignature
from .store import (
    _SCHEMA_VERSION, _append_record, _digest, _identity_reasons,
    _known_hardware, _validate,
)


MAX_RELEASE_BYTES = 8 * 1024 * 1024


def export_release_data(records, destination, *, registry_factory=None,
                        max_bytes=MAX_RELEASE_BYTES):
    """Export explicitly curated records, keeping latest evidence per T20A key.

    The factory takes a SoC string and returns its current descriptor registry.
    This checks identities and measurements, not the truth of a caller's claim
    that they measured real hardware. Release review must establish provenance.
    The destination must be new; a failed export leaves no partial database.
    """
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_RELEASE_BYTES:
        raise ValueError("release budget must be in [1, %d] bytes" % MAX_RELEASE_BYTES)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("release destination must not exist")
    if registry_factory is None:
        from ..runtime.registry import get_global_registry
        registry_factory = lambda soc: get_global_registry()
    def rank(record):
        return (datetime.fromisoformat(record["measurement"]["finished_at"]), record["record_id"])

    latest, registries = {}, {}
    for record in records:
        _validate(record)
        hardware = HardwareSignature(**record["hardware"])
        if (not _known_hardware(hardware)
                or not re.fullmatch(r"[A-Za-z0-9_]+", hardware.soc_model)):
            raise ValueError("release requires a complete HardwareSignature and safe SoC name")
        if hardware.soc_model not in registries:
            registries[hardware.soc_model] = registry_factory(hardware.soc_model)
        identities = tuple((d.kernel_id, d.fingerprint, d.capability.fingerprint)
                           for d in registries[hardware.soc_model].descriptors(op=record["op"]))
        reasons = _identity_reasons(record, hardware, identities)
        if reasons:
            raise ValueError("stale release record %s: %s" % (record["record_id"], "; ".join(reasons)))
        key = (record["op"], _digest(record["bucket"]), _digest(record["hardware"]))
        if key not in latest or rank(record) > rank(latest[key]):
            latest[key] = record

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tuning-release-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        socs = sorted({r["hardware"]["soc_model"] for r in latest.values()})
        # Sequence order also makes the newest measurement win when two
        # different declared buckets overlap at the queried workload.
        for record in sorted(latest.values(), key=rank):
            _append_record(record, staging / record["hardware"]["soc_model"] / "measurements.sqlite3")
        manifest = dict(schema_version=_SCHEMA_VERSION, socs=socs)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        if size > max_bytes:
            raise ValueError("release data is %d bytes; budget is %d bytes" % (size, max_bytes))
        os.replace(staging, destination)
    return dict(records=len(latest), socs=socs, bytes=size)
