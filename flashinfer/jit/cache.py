"""JIT cache identities, pure decisions, and an explicitly used disk store."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import (
    Any, ContextManager, Dict, Iterable, Mapping, Optional, Protocol, Tuple, runtime_checkable,
)

from flashinfer.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    SchemaError,
)

from .core import JitSpec
from .env import JitCompilationPolicy


JIT_CACHE_SCHEMA_VERSION = 1


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if len(str(value)) != 64 or any(
        item not in "0123456789abcdef" for item in str(value)
    ):
        raise SchemaError("JIT cache %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class JitCacheRecord:
    """Verified identity of a compiled artifact already present in a cache."""

    spec_name: str
    spec_fingerprint: str
    environment_fingerprint: str
    artifact: ArtifactRef
    producer_id: str
    build_metadata_fingerprint: str
    schema_version: int = JIT_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_CACHE_SCHEMA_VERSION:
            raise SchemaError("unsupported JIT cache schema version")
        if not self.spec_name or not self.producer_id:
            raise SchemaError("JIT cache spec_name and producer_id must be non-empty")
        for name in (
            "spec_fingerprint",
            "environment_fingerprint",
            "build_metadata_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("JIT cache artifact must be ArtifactRef")
        if self.artifact.kind != ArtifactKind.FILE or self.artifact.format not in {
            ArtifactFormat.ASCENDC_OBJECT,
            ArtifactFormat.ASCENDC_OP_PACKAGE,
            ArtifactFormat.SHARED_LIBRARY,
            ArtifactFormat.TRITON_NPUBIN,
        }:
            raise SchemaError("JIT cache record must identify a compiled file artifact")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "artifact": self.artifact.to_dict(),
            "producer_id": self.producer_id,
            "build_metadata_fingerprint": self.build_metadata_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JitCacheRecord":
        data = dict(value)
        if set(data) != set(cls.__dataclass_fields__):
            raise SchemaError("JitCacheRecord fields are invalid")
        artifact = data.get("artifact")
        if not isinstance(artifact, Mapping):
            raise SchemaError("JIT cache artifact must be an object")
        data["artifact"] = ArtifactRef.from_dict(artifact)
        try:
            return cls(**data)
        except (TypeError, ValueError) as error:
            raise SchemaError("JitCacheRecord fields are invalid") from error

    def validate_spec(self, spec: JitSpec) -> None:
        if (
            self.spec_name != spec.name
            or self.spec_fingerprint != spec.fingerprint
            or self.environment_fingerprint != spec.environment_fingerprint
            or self.artifact.target_soc != spec.target_soc
        ):
            raise SchemaError("JIT cache record does not bind the requested spec")
        if spec.is_compilation_unit and (
            self.artifact.locator != spec.output_path
            or self.artifact.entry_point != spec.entry_points[0]
        ):
            raise SchemaError("JIT cache artifact does not match declared output path and entry point")


@runtime_checkable
class JitBuildStore(Protocol):
    """T18B's disk-cache boundary, independent of the compiler backend.

    The unified orchestrator looks up a validated manifest, acquires the
    per-identity lock on a miss, looks up again under that lock, then builds
    in a private staging root. ``publish`` must atomically publish the complete
    artifact and manifest after verification. Readers must never observe a
    partial build. Lock ownership must survive replacement of the destination.
    Cache roots, staging roots and the read-only package root are separate.

    DiskJitBuildStore implements this contract. JitCacheIndex remains a pure
    in-memory metadata index; importing either creates no cache directories.
    """

    def lookup(self, spec: JitSpec) -> Optional[JitCacheRecord]:
        """Read and validate a published manifest; corruption is a miss."""
        raise NotImplementedError("JIT disk-cache lookup awaits T18B")

    def lock(self, spec: JitSpec) -> ContextManager[None]:
        """Acquire a lock for the full build identity, not a backend-wide lock."""
        raise NotImplementedError("JIT file locking awaits T18B")

    def staging(self, spec: JitSpec) -> ContextManager[Path]:
        """Provide a private root for spec.output_path; clean up on failure."""
        raise NotImplementedError("JIT staging-directory management awaits T18B")

    def publish(
        self, spec: JitSpec, staging_directory: Path, artifact: ArtifactRef,
    ) -> JitCacheRecord:
        """Publish verified bytes and matching manifest atomically under lock."""
        raise NotImplementedError("JIT atomic publication awaits T18B")


class JitCacheIndex:
    """Immutable-identity in-memory index used by Host framework tests."""

    def __init__(self, records: Iterable[JitCacheRecord] = ()) -> None:
        self._records: Dict[str, JitCacheRecord] = {}
        self._paths: Dict[str, str] = {}
        self._lock = RLock()
        for record in records:
            self.publish(record)

    def publish(self, record: JitCacheRecord, *, artifact_path: Optional[Path] = None) -> JitCacheRecord:
        if not isinstance(record, JitCacheRecord):
            raise TypeError("JIT cache accepts JitCacheRecord")
        with self._lock:
            if artifact_path is not None:
                path = str(artifact_path)
            else:
                path = None
            existing = self._records.get(record.spec_name)
            if existing is not None:
                if existing.fingerprint != record.fingerprint:
                    raise SchemaError(
                        "conflicting JIT cache record for %r" % record.spec_name
                    )
                if path is not None:
                    self._paths[record.spec_fingerprint] = path
                return existing
            self._records[record.spec_name] = record
            if path is not None:
                self._paths[record.spec_fingerprint] = path
            return record

    def artifact_path(self, spec: JitSpec) -> Optional[str]:
        """Return an explicitly recorded resolved path, without filesystem I/O."""
        with self._lock:
            return self._paths.get(spec.fingerprint)

    def lookup(self, spec: JitSpec) -> Optional[JitCacheRecord]:
        if not isinstance(spec, JitSpec):
            raise TypeError("JIT cache lookup requires JitSpec")
        with self._lock:
            record = self._records.get(spec.name)
        if record is None:
            return None
        record.validate_spec(spec)
        return record

    @property
    def records(self) -> Tuple[JitCacheRecord, ...]:
        with self._lock:
            values = tuple(self._records.values())
        return tuple(sorted(values, key=lambda item: item.spec_name))


def verify_build_directory(directory: Path, *, spec=None, record=None) -> JitCacheRecord:
    """Same immutable manifest verification for cache and read-only releases.

    This reader owns no cache and writes nothing. Release exports retain the
    exact store manifest; compiler discovery and dependency resolution are absent.
    """
    directory = Path(directory).resolve(strict=True)
    value = json.loads((directory / "manifest.json").read_text())
    if set(value) != {"schema_version", "spec", "files", "record"} or value["schema_version"] != 1:
        raise SchemaError("invalid JIT build manifest schema: manifest.json")
    published_spec = JitSpec.from_dict(value["spec"])
    published = JitCacheRecord.from_dict(value["record"])
    published.validate_spec(published_spec)
    if spec is not None:
        published.validate_spec(spec)
    if record is not None and published != record:
        raise SchemaError("JIT cache record changed before load")
    metadata = {"spec": value["spec"], "files": value["files"]}
    if _canonical_hash(metadata) != published.build_metadata_fingerprint:
        raise SchemaError("JIT build manifest digest mismatch: manifest.json metadata (spec/files)")
    expected = value["files"]
    actual = DiskJitBuildStore._inventory(directory)
    if actual != expected:
        # Releases and the writable cache share this verifier. Preserve the
        # failing path/check so discovery can report corrupt AOT bytes instead
        # of disguising them as a missing artifact or a compilation request.
        if not isinstance(expected, dict):
            raise SchemaError("JIT cache file inventory mismatch: manifest.json files must be an object")
        problems = []
        for name in sorted(set(actual) | set(expected)):
            if name not in actual:
                problems.append("%s: missing file" % name)
            elif name not in expected:
                problems.append("%s: unexpected file" % name)
            elif actual[name] != expected[name]:
                entry = expected[name]
                if not isinstance(entry, dict) or set(entry) != {"digest", "size"}:
                    problems.append("%s: invalid manifest.json inventory entry" % name)
                    continue
                for field in ("digest", "size"):
                    if actual[name][field] != entry[field]:
                        problems.append("%s: %s mismatch (expected %s, got %s)" % (
                            name, field, entry[field], actual[name][field]))
        raise SchemaError("JIT cache file inventory mismatch: " + "; ".join(problems))
    published.artifact.verify_file(directory)
    return published


class DiskJitBuildStore:
    """Local POSIX cache: flock, private staging, immutable generations.

    A published key is an atomically replaced symlink to a complete generation.
    The stable lock lives outside that link. Old generations remain available
    to readers holding a verified path when a corrupt key is rebuilt.
    Construction performs no I/O. Cleanup/eviction is deliberately explicit.
    """

    def __init__(self, root: Optional[Path] = None, *, namespace: Optional[str] = None) -> None:
        base = Path(root) if root is not None else Path(os.environ.get(
            "FLASHINFER_NPU_CACHE_DIR", "~/.cache/flashinfer")).expanduser()
        self.root = base.absolute() / "jit"
        if namespace is not None:
            if not namespace or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in namespace):
                raise SchemaError("invalid user JIT cache namespace")
            self.root = self.root / "user" / namespace
        self._directories: Dict[str, Path] = {}
        self._mutex = RLock()

    def _key_path(self, fingerprint: str) -> Path:
        return self.root / fingerprint[:2] / fingerprint

    def _unit(self, spec: JitSpec) -> None:
        if not isinstance(spec, JitSpec) or not spec.is_compilation_unit:
            raise SchemaError("disk builds require a JIT compilation unit")
        reserved = {"manifest.json", "build.log", "request.json", "source.py", "triton.json", "triton-cache"}
        if Path(spec.output_path).parts[0] in reserved:
            raise SchemaError("output_path collides with a reserved build path")

    @staticmethod
    def _inventory(directory: Path) -> Dict[str, Any]:
        result = {}
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise SchemaError("build output must not contain symlinks: %s" % path)
            if path.is_file() and path not in {directory / "manifest.json", directory / "build.log"}:
                payload = path.read_bytes()
                result[path.relative_to(directory).as_posix()] = {
                    "digest": hashlib.sha256(payload).hexdigest(), "size": len(payload),
                }
        return result

    def _read(self, directory: Path, *, spec: Optional[JitSpec] = None,
              record: Optional[JitCacheRecord] = None) -> JitCacheRecord:
        directory = directory.resolve(strict=True)
        if not directory.is_relative_to((self.root / ".objects").resolve()):
            raise SchemaError("published JIT cache generation escapes object root")
        published = verify_build_directory(directory, spec=spec, record=record)
        with self._mutex:
            self._directories[published.fingerprint] = directory
        return published

    def lookup(self, spec: JitSpec) -> Optional[JitCacheRecord]:
        self._unit(spec)
        try:
            return self._read(self._key_path(spec.fingerprint), spec=spec)
        except (OSError, ValueError, TypeError, KeyError):
            # A corrupt/missing published manifest is a miss by the 18A contract.
            return None

    def directory(self, record: JitCacheRecord) -> Path:
        with self._mutex:
            directory = self._directories.get(record.fingerprint)
        if directory is None:
            directory = self._key_path(record.spec_fingerprint).resolve(strict=True)
        self._read(directory, record=record)
        return directory

    def artifact_path(self, record: JitCacheRecord) -> Path:
        return self.directory(record) / record.artifact.locator

    @contextmanager
    def lock(self, spec: JitSpec):
        import fcntl

        self._unit(spec)
        directory = self.root / ".locks"
        directory.mkdir(parents=True, exist_ok=True)
        # Never unlink this inode; replacement of a published key cannot split it.
        with (directory / (spec.fingerprint + ".lock")).open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def staging(self, spec: JitSpec):
        self._unit(spec)
        parent = self.root / ".staging"
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=spec.fingerprint + "-", dir=parent))
        try:
            yield directory
        except BaseException as error:
            source = directory / "build.log"
            if source.is_file():
                logs = self.root / ".logs"
                logs.mkdir(parents=True, exist_ok=True)
                log_path = logs / (directory.name + ".log")
                shutil.copyfile(source, log_path)
                from .errors import JitBuildError

                if isinstance(error, JitBuildError):
                    error.log_path = log_path
            raise
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def publish(self, spec: JitSpec, staging_directory: Path,
                artifact: ArtifactRef) -> JitCacheRecord:
        self._unit(spec)
        directory = Path(staging_directory).resolve(strict=True)
        if directory.parent != (self.root / ".staging").resolve():
            raise SchemaError("publication requires this store's private staging directory")
        artifact.verify_file(directory)
        files = self._inventory(directory)
        metadata = {"spec": spec.to_dict(), "files": files}
        record = JitCacheRecord(spec.name, spec.fingerprint, spec.environment_fingerprint,
            artifact, spec.generator_id, _canonical_hash(metadata))
        record.validate_spec(spec)
        manifest = dict(schema_version=1, record=record.to_dict(), **metadata)
        (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        # Flush the whole generation before exposing its pointer.
        for path in directory.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for path in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
            self._fsync_directory(path)
        objects = self.root / ".objects"
        objects.mkdir(parents=True, exist_ok=True)
        generation = objects / (spec.fingerprint + "-" + uuid.uuid4().hex)
        os.rename(directory, generation)
        self._fsync_directory(generation)
        self._fsync_directory(objects)
        target = self._key_path(spec.fingerprint)
        target.parent.mkdir(parents=True, exist_ok=True)
        pointer = target.parent / (".publish-" + uuid.uuid4().hex)
        try:
            pointer.symlink_to(os.path.relpath(generation, target.parent), target_is_directory=True)
            # A foreign/legacy directory is never a valid generation pointer.
            # Quarantine it under the key lock before installing the new link.
            if target.is_dir() and not target.is_symlink():
                os.rename(target, objects / ("quarantine-" + uuid.uuid4().hex))
            os.replace(pointer, target)
            self._fsync_directory(target.parent)
            self._fsync_directory(self.root)
        finally:
            pointer.unlink(missing_ok=True)
        with self._mutex:
            self._directories[record.fingerprint] = generation
        return record

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


jit_cache_index = JitCacheIndex()


class JitResolutionState(str, Enum):
    CACHE_HIT = "cache_hit"
    BUILD_REQUIRED = "build_required"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class JitResolution:
    spec_name: str
    spec_fingerprint: str
    state: JitResolutionState
    reason: str
    cache_record_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.spec_name or not self.reason:
            raise SchemaError("JIT resolution spec_name and reason must be non-empty")
        _require_hash("resolution spec_fingerprint", self.spec_fingerprint)
        object.__setattr__(self, "state", JitResolutionState(self.state))
        if self.state == JitResolutionState.CACHE_HIT:
            if self.cache_record_fingerprint is None:
                raise SchemaError("JIT cache hit requires a cache record fingerprint")
            _require_hash(
                "resolution cache_record_fingerprint",
                self.cache_record_fingerprint,
            )
        elif self.cache_record_fingerprint is not None:
            raise SchemaError("JIT cache miss cannot identify a cache record")

    @property
    def ready(self) -> bool:
        return self.state == JitResolutionState.CACHE_HIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "state": self.state.value,
            "reason": self.reason,
            "cache_record_fingerprint": self.cache_record_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


class MissingJITCacheError(RuntimeError):
    """Raised when the policy forbids satisfying a missing JIT artifact."""


def resolve_jit_spec(
    spec: JitSpec,
    cache: JitCacheIndex,
    policy: JitCompilationPolicy,
) -> JitResolution:
    """Return a pure decision; never compile, load, or touch the filesystem."""

    if not isinstance(spec, JitSpec):
        raise TypeError("resolve_jit_spec requires JitSpec")
    if not isinstance(cache, JitCacheIndex):
        raise TypeError("resolve_jit_spec requires JitCacheIndex")
    policy = JitCompilationPolicy(policy)
    record = cache.lookup(spec)
    if record is not None:
        return JitResolution(
            spec.name,
            spec.fingerprint,
            JitResolutionState.CACHE_HIT,
            "verified cache record matches spec and environment",
            record.fingerprint,
        )
    if policy == JitCompilationPolicy.ENABLED:
        return JitResolution(
            spec.name,
            spec.fingerprint,
            JitResolutionState.BUILD_REQUIRED,
            "cache miss requires an authorized build provider",
        )
    return JitResolution(
        spec.name,
        spec.fingerprint,
        JitResolutionState.UNAVAILABLE,
        "JIT cache miss and compilation policy is %s" % policy.value,
    )


def require_jit_cache_hit(resolution: JitResolution) -> None:
    if not isinstance(resolution, JitResolution):
        raise TypeError("resolution must be JitResolution")
    if not resolution.ready:
        raise MissingJITCacheError(resolution.reason)


__all__ = [
    "JIT_CACHE_SCHEMA_VERSION",
    "JitCacheIndex",
    "DiskJitBuildStore",
    "jit_cache_index",
    "JitBuildStore",
    "JitCacheRecord",
    "JitResolution",
    "JitResolutionState",
    "MissingJITCacheError",
    "require_jit_cache_hit",
    "resolve_jit_spec",
]
