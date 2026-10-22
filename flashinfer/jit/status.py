"""Read-only kernel / declared output / entry-symbol mappings.

This module never invokes a backend, payload reader or module loader, and
never probes files. ``is_compiled=True`` means a compiled ArtifactRef was
recorded, not that bytes, file existence, ABI or resolved symbols were checked
by this query. Missing execution-body metadata is explicitly unknown (None).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from flashinfer.runtime import ArtifactKind, ArtifactRef, Backend, KernelRegistry, SourceRef

from .cache import JitCacheIndex, jit_cache_index
from .core import JitSpecRegistry, jit_spec_registry


class JitArtifactState(str, Enum):
    UNKNOWN_KERNEL = "unknown_kernel"
    UNDECLARED = "undeclared"
    SOURCE_ONLY = "source_only"
    DECLARED = "declared"
    ARTIFACT_RECORDED = "artifact_recorded"
    BUILTIN = "builtin"


@dataclass(frozen=True)
class JitArtifactStatus:
    """One mapping, with expected output separate from recorded ArtifactRef.

    A kernel can have several declarations (specializations or environments).
    Each gets its own row; a descriptor's artifact is a separate row unless an
    exact cache record already binds it to a declaration. No arbitrary variant
    is selected, and no compilation is inferred from a recipe or filename.
    """

    kernel_id: str
    known_kernel: bool
    backend: Optional[Backend]
    state: JitArtifactState
    is_compiled: Optional[bool]
    source: Optional[SourceRef]
    target_soc: Optional[str]
    artifact: Optional[ArtifactRef]
    artifact_path: Optional[str]
    expected_artifact_path: Optional[str]
    entry_point: Optional[str]
    compile_options: Tuple[str, ...]
    spec_name: Optional[str]
    spec_fingerprint: Optional[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "known_kernel": self.known_kernel,
            "backend": self.backend.value if self.backend is not None else None,
            "state": self.state.value,
            "is_compiled": self.is_compiled,
            "source": self.source.to_dict() if self.source is not None else None,
            "target_soc": self.target_soc,
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "artifact_path": self.artifact_path,
            "expected_artifact_path": self.expected_artifact_path,
            "entry_point": self.entry_point,
            "compile_options": list(self.compile_options),
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "reason": self.reason,
        }


def describe_artifact(
    kernel_id: str,
    *,
    registry: Optional[KernelRegistry] = None,
    spec_registry: Optional[JitSpecRegistry] = None,
    cache: Optional[JitCacheIndex] = None,
) -> Tuple[JitArtifactStatus, ...]:
    """Return every known mapping for a kernel, without compilation or loading.

    The result always has at least one row, even for an unknown kernel or a
    legacy descriptor without SourceRef. All roots and file existence remain
    unobserved; artifact paths are the schema's relative locators. An optional
    in-memory cache contributes only records bound to an exact spec identity.
    """
    if registry is None:
        from flashinfer.runtime.registry import get_global_registry

        registry = get_global_registry()
    if spec_registry is None:
        spec_registry = jit_spec_registry
    if cache is None:
        cache = jit_cache_index
    if cache is not None and not isinstance(cache, JitCacheIndex):
        raise TypeError("cache must be JitCacheIndex or None")
    try:
        descriptor = registry.get(kernel_id)
    except KeyError:
        return (JitArtifactStatus(
            kernel_id, False, None, JitArtifactState.UNKNOWN_KERNEL, None,
            None, None, None, None, None, None, (), None, None,
            "kernel_id is not registered",
        ),)

    statuses = []
    for spec in spec_registry.specs_for_kernel(kernel_id):
        spec.validate_descriptor(descriptor)
        record = cache.lookup(spec) if cache is not None else None
        artifact = record.artifact if record is not None else None
        statuses.append(JitArtifactStatus(
            kernel_id, True, descriptor.backend,
            JitArtifactState.ARTIFACT_RECORDED if artifact is not None else JitArtifactState.DECLARED,
            artifact is not None, spec.source, spec.target_soc, artifact,
            (cache.artifact_path(spec) or artifact.locator) if artifact is not None else None,
            spec.output_path, spec.entry_points[0], spec.compile_options,
            spec.name, spec.fingerprint,
            ("compiled artifact identity recorded for this spec; filesystem and symbols not checked"
             if artifact is not None else "compilation unit declared; no compiled artifact recorded"),
        ))

    artifact = descriptor.artifact
    # A descriptor artifact without a cache binding is not attributed to any
    # particular specialization, even if its path resembles a declared output.
    if statuses and (artifact is None or any(s.artifact == artifact for s in statuses)):
        return tuple(statuses)

    entry = artifact.entry_point if artifact is not None else None
    if entry is None and descriptor.launch_abi is not None:
        entry = descriptor.launch_abi.entry_point
    target = artifact.target_soc if artifact is not None else descriptor.target_soc
    if artifact is not None and artifact.kind == ArtifactKind.FILE:
        state, compiled = JitArtifactState.ARTIFACT_RECORDED, True
        reason = "descriptor records a compiled artifact identity; filesystem and symbols not checked"
    elif artifact is not None and artifact.kind == ArtifactKind.BUILTIN:
        state, compiled = JitArtifactState.BUILTIN, None
        reason = "builtin provider; no local compilation status"
    elif descriptor.source is not None or artifact is not None:
        state, compiled = JitArtifactState.SOURCE_ONLY, False
        reason = "source identity recorded; descriptor has no compiled artifact"
    else:
        state, compiled = JitArtifactState.UNDECLARED, None
        reason = "no source or artifact declared; compilation status is unknown"
    statuses.append(JitArtifactStatus(
        kernel_id, True, descriptor.backend, state, compiled, descriptor.source,
        target, artifact,
        artifact.locator if artifact is not None and artifact.kind == ArtifactKind.FILE else None,
        None, entry, (), None, None, reason,
    ))
    return tuple(statuses)


__all__ = ["JitArtifactState", "JitArtifactStatus", "describe_artifact"]
