"""Metadata-only execution identities; no provider imports or compiler probes."""

from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=8)
def execution_sources(paths):
    root = Path(__file__).resolve().parents[1]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


@lru_cache(maxsize=1)
def runtime_identity():
    """Process snapshot. Package replacements require a new process."""
    versions = {}
    for name in ("torch", "torch-npu", "triton"):
        try:
            versions[name.replace("-", "_")] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name.replace("-", "_")] = "missing"
    root = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    versions["cann"] = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                        for relative in ("share/info/runtime/version.info", "share/info/version.info")
                        if (root / relative).is_file()}
    versions["python_abi"] = sys.implementation.cache_tag
    return versions


def candidate_identity(descriptor, signature, hardware):
    identity = dict(descriptor_fingerprint=descriptor.fingerprint,
                    capability_fingerprint=descriptor.capability.fingerprint)
    provider = getattr(descriptor.runner, "tuning_identity", None)
    if provider is not None:
        identity["execution_fingerprint"] = provider(signature, hardware)
    return identity
