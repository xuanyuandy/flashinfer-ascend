"""Lazy discovery of immutable, shipped Ascend C op-package releases.

The process index caches only directory listings and verified manifest metadata.
Root inode/link-count/mtime/ctime changes invalidate listings; manifest inode/mtime/ctime/
size changes invalidate metadata. Changing FLASHINFER_AOT_PATHS selects a new
index. ``invalidate_op_package_index`` (or ``index.invalidate``) explicitly
refreshes it, including unusual deployments which preserve file timestamps.
Selected packages are fully verified again on every find/load, so the index is
never evidence that payload bytes are still intact. No disk index is written.

The default root is the installed wheel's flashinfer/data/op_packages directory.
FLASHINFER_AOT_PATHS adds os.pathsep-separated roots (or individual releases).
Only immediate package directories are indexed, never recursively searched.
Importing this module performs no filesystem, compiler, or device discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Optional, Tuple

from flashinfer.runtime import ArtifactFormat, KernelDescriptor

from .cache import JitCacheRecord, verify_build_directory
from .core import JitSpec
from .op_package import OpPackageSymbolError, load_op_package, verify_op_package
from .recipe import BuildRecipe


class AotDiscoveryError(RuntimeError):
    """Base for runtime AOT diagnostics; only AotMissingError allows fallback."""


@dataclass(frozen=True)
class AotIncompatibleCandidate:
    directory: Path
    reasons: Tuple[str, ...]
    category: str = "incompatible"


class AotMissingError(AotDiscoveryError):
    category = "missing"

    def __init__(self, spec, target_soc, roots, rejections=(), *, detail=None):
        self.spec = spec
        self.target_soc = target_soc
        self.roots = tuple(roots)
        self.rejections = tuple(rejections)
        self.candidate_count = len(self.rejections)
        message = (
            "no matching AOT op-package for kernel_id=%s, target_soc=%s, spec=%s; "
            "searched paths=%s"
            % (spec.kernel_id, target_soc, spec.fingerprint,
               ", ".join(str(root) for root in self.roots) or "<none>")
        )
        if self.rejections:
            message += "; found %d candidate(s), all incompatible: %s" % (
                self.candidate_count,
                "; ".join("%s: %s" % (item.directory, ", ".join(item.reasons))
                          for item in self.rejections),
            )
        else:
            message += "; found 0 candidates"
        if detail:
            message += "; " + detail
        super().__init__(message)


class AotCorruptError(AotDiscoveryError):
    category = "corrupt"

    def __init__(self, directory, error):
        self.directory = Path(directory)
        super().__init__("corrupt AOT op-package %s: %s: %s" % (
            self.directory, type(error).__name__, error))


class AotLoadError(AotDiscoveryError):
    category = "load error"

    def __init__(self, directory, error):
        self.directory = Path(directory)
        super().__init__("AOT op-package load error at %s: %s: %s" % (
            self.directory, type(error).__name__, error))


@dataclass(frozen=True)
class AotPackageMatch:
    directory: Path
    spec: JitSpec
    record: JitCacheRecord
    recipe: BuildRecipe

    def load(self):
        """Use the existing loader; corruption and activation failures are fatal.

        CANN registrations are process-scoped. The existing loader enforces a
        single record identity per process; its fresh-process diagnostic and
        original exception are preserved as AotLoadError. Multiple identities
        are deliberately not attempted by discovery after a load failure.

        Known gap: neither this matcher nor load_op_package compares the
        currently installed CANN version against the build environment. Exact
        recorded spec/ABI equality does not establish runtime CANN compatibility.
        """
        _verify(self.directory, spec=self.spec, record=self.record)
        try:
            return load_op_package(self.directory, target_soc=self.spec.target_soc)
        except Exception as error:
            raise AotLoadError(self.directory, error) from error


def op_package_search_paths():
    """Installed layout first, then explicitly configured extra search roots."""
    default = Path(__file__).resolve().parents[1] / "data" / "op_packages"
    extra = (Path(value).expanduser() for value in
             os.environ.get("FLASHINFER_AOT_PATHS", "").split(os.pathsep) if value)
    return tuple(dict.fromkeys(path.absolute() for path in (default, *extra)))


def _stamp(path):
    try:
        value = path.stat()
    except FileNotFoundError:
        return None
    return (value.st_dev, value.st_ino, value.st_nlink, value.st_mtime_ns,
            value.st_ctime_ns, value.st_size)


def _verify(directory, *, spec=None, record=None):
    """Delegate all trust decisions to the store and op-package verifiers."""
    try:
        published = verify_build_directory(directory, spec=spec, record=record)
        value = json.loads((directory / "manifest.json").read_text())
        published_spec = JitSpec.from_dict(value["spec"])
        recipe = None
        if published.artifact.format == ArtifactFormat.ASCENDC_OP_PACKAGE:
            _, recipe = verify_op_package(directory, published_spec)
        return published_spec, published, recipe
    except OpPackageSymbolError as error:
        raise AotLoadError(directory, error) from error
    except Exception as error:
        raise AotCorruptError(directory, error) from error


@dataclass(frozen=True)
class _IndexedCandidate:
    directory: Path
    stamp: object
    spec: Optional[JitSpec] = None
    record: Optional[JitCacheRecord] = None
    recipe: Optional[BuildRecipe] = None
    error: Optional[AotCorruptError] = None


class AotPackageIndex:
    """A process-local metadata index with explicit and automatic invalidation.

    Release directories must be immutable after publication. Root entry and
    manifest changes are observed automatically; call invalidate after unusual
    in-place deployments or before intentionally replacing an active runner.
    Even with an unchanged index, a selected payload is never trusted without
    the existing verifier's complete inventory and digest checks.
    """

    def __init__(self, roots=None):
        self.roots = tuple(dict.fromkeys(Path(root).expanduser().absolute()
                          for root in (op_package_search_paths() if roots is None else roots)))
        self._root_stamps = None
        self._entries = {}
        self._lock = RLock()

    def invalidate(self):
        with self._lock:
            self._root_stamps = None
            self._entries.clear()

    def _directories(self):
        directories = []
        for root in self.roots:
            if not root.exists():
                continue
            if ((root / "manifest.json").exists()
                    or (root / "op-package.json").exists()
                    or re.fullmatch(r"[0-9a-f]{64}", root.name)):
                directories.append(root)
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                # A missing manifest in a SHA-addressed release is corruption,
                # not a silent cache miss. Ignore unrelated ancillary folders.
                if (re.fullmatch(r"[0-9a-f]{64}", child.name)
                        or (child / "manifest.json").exists()
                        or (child / "op-package.json").exists()):
                    directories.append(child)
        return tuple(dict.fromkeys(directories))

    def _snapshot(self):
        with self._lock:
            try:
                root_stamps = tuple(_stamp(root) for root in self.roots)
                if self._root_stamps != root_stamps:
                    directories = self._directories()
                else:
                    directories = tuple(self._entries)
                entries = {}
                for directory in directories:
                    stamp = _stamp(directory / "manifest.json")
                    entry = self._entries.get(directory)
                    if entry is None or entry.stamp != stamp:
                        try:
                            spec, record, recipe = _verify(directory)
                            entry = _IndexedCandidate(directory, stamp, spec, record, recipe)
                        except AotCorruptError as error:
                            entry = _IndexedCandidate(directory, stamp, error=error)
                    entries[directory] = entry
                self._entries = entries
                self._root_stamps = root_stamps
                return tuple(entries.values())
            except OSError as error:
                raise AotCorruptError(getattr(error, "filename", None) or "<search roots>", error) from error

    def find(self, spec, *, descriptor, target_soc):
        """Match recorded spec, requested SoC and the complete descriptor ABI.

        The existing spec's ``kernel`` input fingerprint binds the complete
        descriptor, including launch and binary ABIs, without inventing a second
        serialized ABI contract. The caller supplies the trusted descriptor used
        to declare that spec; runners do not participate in its fingerprint.
        """
        if not isinstance(spec, JitSpec) or not isinstance(descriptor, KernelDescriptor):
            raise TypeError("AOT discovery requires JitSpec and KernelDescriptor")
        entries = self._snapshot()
        # A malformed manifest cannot reliably tell us its kernel identity.
        # Fail closed instead of reclassifying it as an unrelated/missing file.
        for entry in entries:
            if entry.error is not None:
                raise entry.error
        rejections = []
        for entry in entries:
            reasons = []
            if entry.record.artifact.format != ArtifactFormat.ASCENDC_OP_PACKAGE:
                reasons.append("artifact format is %s, expected ascendc_op_package" %
                               entry.record.artifact.format.value)
            if entry.spec.target_soc != target_soc:
                reasons.append("target_soc mismatch: package=%s, requested=%s" %
                               (entry.spec.target_soc, target_soc))
            if entry.spec.fingerprint != spec.fingerprint:
                reasons.append("spec fingerprint mismatch: package=%s, requested=%s" %
                               (entry.spec.fingerprint, spec.fingerprint))
            if dict(entry.spec.input_fingerprints).get("kernel") != descriptor.fingerprint:
                reasons.append("descriptor/ABI fingerprint mismatch: package=%s, requested=%s" %
                               (dict(entry.spec.input_fingerprints).get("kernel"), descriptor.fingerprint))
            try:
                entry.spec.validate_descriptor(descriptor)
                if entry.recipe is not None:
                    entry.recipe.validate_descriptor(descriptor)
            except (TypeError, ValueError) as error:
                reasons.append("descriptor/ABI incompatible: %s" % error)
            if reasons:
                rejections.append(AotIncompatibleCandidate(entry.directory, tuple(reasons)))
                continue
            # The index never substitutes for verifying current payload bytes.
            published_spec, record, recipe = _verify(entry.directory, spec=spec, record=entry.record)
            return AotPackageMatch(entry.directory, published_spec, record, recipe)
        raise AotMissingError(spec, target_soc, self.roots, rejections)


_indexes = {}
_indexes_lock = RLock()


def get_op_package_index():
    roots = op_package_search_paths()
    with _indexes_lock:
        if roots not in _indexes:
            _indexes[roots] = AotPackageIndex(roots)
        return _indexes[roots]


def invalidate_op_package_index():
    """Invalidate all process indexes, including callers retaining an index."""
    with _indexes_lock:
        for index in _indexes.values():
            index.invalidate()
        _indexes.clear()


__all__ = [
    "AotDiscoveryError", "AotMissingError", "AotCorruptError", "AotLoadError",
    "AotIncompatibleCandidate", "AotPackageMatch", "AotPackageIndex",
    "op_package_search_paths", "get_op_package_index", "invalidate_op_package_index",
]
