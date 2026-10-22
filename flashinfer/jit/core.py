"""Framework-only JIT specifications and registry.

This mirrors the responsibility boundary of ``flashinfer.jit.core`` without
claiming that an Ascend compiler, loader, or executable kernel exists.  Specs
are immutable build identities; compilation is intentionally outside this
checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from flashinfer.runtime import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    KernelDescriptor,
    SchemaError,
    SourceRef,
)


JIT_SPEC_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_ENTRY_POINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(name: str, value: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise SchemaError("invalid JIT %s" % name)
    return result


def _sha256(name: str, value: str) -> str:
    result = str(value)
    if not _SHA256.fullmatch(result):
        raise SchemaError("JIT %s must be lowercase SHA-256" % name)
    return result


@dataclass(frozen=True)
class JitSpec:
    """Canonical recipe identity for one generated module.

    T18A compilation units bind ``kernel_id``, ``source`` (the runtime's
    SourceRef), ``output_path`` and exactly one of ``entry_points``. The output
    path is relative to an artifact root, not evidence that a file exists.
    Only an actual build can produce an ArtifactRef with a digest and size.

    The older Attention recipe form (including ``source_artifacts``) remains
    supported with its original serialization and fingerprint. It is not a
    kernel compilation unit until the three new fields are supplied. Neither
    form reads source bytes or probes a compiler.
    """

    name: str
    domain: str
    generator_id: str
    generator_version: str
    target_soc: str
    environment_fingerprint: str
    specialization_fingerprint: str
    input_fingerprints: Tuple[Tuple[str, str], ...] = ()
    source_artifacts: Tuple[ArtifactRef, ...] = ()
    compile_options: Tuple[str, ...] = ()
    entry_points: Tuple[str, ...] = ()
    schema_version: int = JIT_SPEC_SCHEMA_VERSION
    kernel_id: Optional[str] = None
    source: Optional[SourceRef] = None
    output_path: Optional[str] = None

    def __post_init__(self) -> None:
        if any(item is not None for item in (self.kernel_id, self.source, self.output_path)):
            if (not isinstance(self.target_soc, str) or not self.target_soc
                    or any(c.isspace() or c == "\0" for c in self.target_soc)):
                raise SchemaError("JIT compilation target_soc must be a SOC identifier")
            if (not isinstance(self.compile_options, (tuple, list))
                    or any(not isinstance(item, str) for item in self.compile_options)):
                raise SchemaError("JIT compile options must be a list or tuple of strings")
            if (not isinstance(self.entry_points, (tuple, list))
                    or any(not isinstance(item, str) for item in self.entry_points)):
                raise SchemaError("JIT entry points must be a list or tuple of symbols")
        if self.schema_version != JIT_SPEC_SCHEMA_VERSION:
            raise SchemaError("unsupported JIT spec schema version")
        for name in ("name", "domain", "generator_id"):
            object.__setattr__(self, name, _identifier(name, getattr(self, name)))
        for name in ("generator_version", "target_soc"):
            value = str(getattr(self, name))
            if not value or any(item in value for item in ("\x00", "\n", "\r")):
                raise SchemaError("JIT %s must be a safe non-empty string" % name)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "environment_fingerprint",
            _sha256("environment_fingerprint", self.environment_fingerprint),
        )
        object.__setattr__(
            self,
            "specialization_fingerprint",
            _sha256("specialization_fingerprint", self.specialization_fingerprint),
        )

        bindings = tuple(
            (str(name), str(digest)) for name, digest in self.input_fingerprints
        )
        if any(not _IDENTIFIER.fullmatch(name) for name, _ in bindings):
            raise SchemaError("JIT input fingerprint names are invalid")
        if any(not _SHA256.fullmatch(digest) for _, digest in bindings):
            raise SchemaError("JIT input fingerprints must be lowercase SHA-256")
        if len({name for name, _ in bindings}) != len(bindings):
            raise SchemaError("JIT input fingerprint names must be unique")
        object.__setattr__(self, "input_fingerprints", tuple(sorted(bindings)))

        sources = tuple(self.source_artifacts)
        if any(not isinstance(item, ArtifactRef) for item in sources):
            raise TypeError("JIT source_artifacts must contain ArtifactRef")
        if any(
            item.kind != ArtifactKind.JIT_SOURCE
            or item.format != ArtifactFormat.ASCENDC_SOURCE
            for item in sources
        ):
            raise SchemaError("JIT source artifacts must be Ascend C source identities")
        if len({item.locator for item in sources}) != len(sources):
            raise SchemaError("JIT source artifact locators must be unique")
        object.__setattr__(
            self,
            "source_artifacts",
            tuple(sorted(sources, key=lambda item: item.locator)),
        )

        options = tuple(str(item) for item in self.compile_options)
        if any(
            not item
            or any(character in item for character in ("\x00", "\n", "\r"))
            for item in options
        ):
            raise SchemaError("JIT compile options must be safe non-empty strings")
        object.__setattr__(self, "compile_options", options)

        entries = tuple(str(item) for item in self.entry_points)
        if any(not _ENTRY_POINT.fullmatch(item) for item in entries):
            raise SchemaError("JIT entry points are invalid")
        if len(set(entries)) != len(entries):
            raise SchemaError("JIT entry points must be unique")
        object.__setattr__(self, "entry_points", entries)

        declaration = (self.kernel_id, self.source, self.output_path)
        if any(item is not None for item in declaration):
            if any(item is None for item in declaration):
                raise SchemaError("JIT compilation unit requires kernel_id, source and output_path")
            if (not isinstance(self.kernel_id, str) or not self.kernel_id
                    or any(c.isspace() or c == "\0" for c in self.kernel_id)):
                raise SchemaError("JIT kernel_id must be a non-empty identifier")
            if not isinstance(self.source, SourceRef):
                raise SchemaError("JIT source must be SourceRef")
            if self.source_artifacts:
                raise SchemaError("JIT compilation unit uses SourceRef, not legacy source_artifacts")
            if not isinstance(self.output_path, str) or not self.output_path:
                raise SchemaError("JIT output_path must be a normalized relative POSIX path")
            path = PurePosixPath(self.output_path)
            if (path.is_absolute() or ".." in path.parts or self.output_path == "."
                    or str(path) != self.output_path or "\\" in self.output_path
                    or any(c in self.output_path for c in ("\0", "\n", "\r"))):
                raise SchemaError("JIT output_path must be a normalized relative POSIX path")
            if len(entries) != 1:
                raise SchemaError("JIT compilation unit requires exactly one entry point")

    @property
    def is_compilation_unit(self) -> bool:
        return self.kernel_id is not None

    def validate_descriptor(self, descriptor: KernelDescriptor) -> None:
        """Bind a unit to the registry without changing it or its dispatch identity."""
        if not isinstance(descriptor, KernelDescriptor):
            raise TypeError("descriptor must be KernelDescriptor")
        if not self.is_compilation_unit:
            raise SchemaError("legacy JIT recipe has no kernel compilation-unit binding")
        target = (descriptor.artifact.target_soc if descriptor.artifact is not None
                  else descriptor.target_soc)
        if (self.kernel_id != descriptor.kernel_id or self.source != descriptor.source
                or (target is not None and self.target_soc != target)):
            raise SchemaError("JIT compilation unit does not bind kernel source and target_soc")
        entry = (descriptor.artifact.entry_point if descriptor.artifact is not None
                 else None)
        if entry is None and descriptor.launch_abi is not None:
            entry = descriptor.launch_abi.entry_point
        if entry is not None and self.entry_points != (entry,):
            raise SchemaError("JIT entry point does not match kernel execution body")

    @property
    def source_materialized(self) -> bool:
        """Explicit source identity recorded; this does not probe its bytes."""
        return self.source is not None or bool(self.source_artifacts)

    def to_dict(self) -> Dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "name": self.name,
            "domain": self.domain,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "target_soc": self.target_soc,
            "environment_fingerprint": self.environment_fingerprint,
            "specialization_fingerprint": self.specialization_fingerprint,
            "input_fingerprints": [list(item) for item in self.input_fingerprints],
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
            "compile_options": list(self.compile_options),
            "entry_points": list(self.entry_points),
        }
        if self.is_compilation_unit:
            value.update(kernel_id=self.kernel_id, source=self.source.to_dict(),
                         output_path=self.output_path)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JitSpec":
        data = dict(value)
        unit_fields = {"kernel_id", "source", "output_path"}
        legacy_fields = set(cls.__dataclass_fields__) - unit_fields
        if set(data) not in (legacy_fields, set(cls.__dataclass_fields__)):
            raise SchemaError("JitSpec fields are invalid")
        try:
            data["input_fingerprints"] = tuple(
                tuple(item) for item in data["input_fingerprints"]
            )
            data["source_artifacts"] = tuple(
                ArtifactRef.from_dict(item) for item in data["source_artifacts"]
            )
            # __post_init__ canonicalizes arrays after validating the new unit
            # form; converting strings here would conceal malformed JSON.
            if "source" in data and data["source"] is not None:
                data["source"] = SourceRef.from_dict(data["source"])
            return cls(**data)
        except (KeyError, TypeError, ValueError) as error:
            raise SchemaError("JitSpec fields are invalid") from error

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class JitSpecStatus:
    """Deterministic registration status without filesystem probing."""

    name: str
    spec_fingerprint: str
    registration_generation: int
    source_materialized: bool

    def __post_init__(self) -> None:
        _identifier("status name", self.name)
        _sha256("status spec_fingerprint", self.spec_fingerprint)
        if self.registration_generation < 1:
            raise SchemaError("JIT registration generation must be positive")


class JitSpecRegistry:
    """Explicit registry for immutable JIT recipes.

    Re-registering an identical spec is idempotent.  Reusing a name for a
    different identity fails closed, so a wrapper cannot silently change the
    recipe behind an already planned module.
    """

    def __init__(self, specs: Iterable[JitSpec] = ()) -> None:
        self._specs: Dict[str, JitSpec] = {}
        self._generations: Dict[str, int] = {}
        self._generation = 0
        self._lock = RLock()
        for spec in specs:
            self.register(spec)

    def register(self, spec: JitSpec) -> JitSpec:
        if not isinstance(spec, JitSpec):
            raise TypeError("JIT registry accepts JitSpec")
        with self._lock:
            existing = self._specs.get(spec.name)
            if existing is not None:
                if existing.fingerprint != spec.fingerprint:
                    raise SchemaError("conflicting JIT spec name %r" % spec.name)
                return existing
            self._generation += 1
            self._specs[spec.name] = spec
            self._generations[spec.name] = self._generation
            return spec

    def get(self, name: str) -> Optional[JitSpec]:
        with self._lock:
            return self._specs.get(str(name))

    def get_all_specs(self) -> Dict[str, JitSpec]:
        with self._lock:
            return dict(self._specs)

    def specs(self, domain: Optional[str] = None) -> Tuple[JitSpec, ...]:
        with self._lock:
            values = tuple(self._specs.values())
        if domain is not None:
            values = tuple(item for item in values if item.domain == domain)
        return tuple(sorted(values, key=lambda item: item.name))

    def specs_for_kernel(self, kernel_id: str) -> Tuple[JitSpec, ...]:
        """All declared variants; never guess a kernel binding for old recipes."""
        return tuple(spec for spec in self.specs() if spec.kernel_id == kernel_id)

    def get_spec_status(self, name: str) -> Optional[JitSpecStatus]:
        with self._lock:
            spec = self._specs.get(str(name))
            if spec is None:
                return None
            generation = self._generations[spec.name]
            return JitSpecStatus(
                name=spec.name,
                spec_fingerprint=spec.fingerprint,
                registration_generation=generation,
                source_materialized=spec.source_materialized,
            )

    def get_all_statuses(self) -> Tuple[JitSpecStatus, ...]:
        statuses = []
        for spec in self.specs():
            status = self.get_spec_status(spec.name)
            if status is not None:
                statuses.append(status)
        return tuple(statuses)

    def get_stats(self) -> Dict[str, int]:
        statuses = self.get_all_statuses()
        materialized = sum(1 for item in statuses if item.source_materialized)
        return {
            "total": len(statuses),
            "source_materialized": materialized,
            "recipe_only": len(statuses) - materialized,
        }


jit_spec_registry = JitSpecRegistry()


def gen_jit_spec(
    *,
    name: str,
    domain: str,
    generator_id: str,
    generator_version: str,
    target_soc: str,
    environment_fingerprint: str,
    specialization_fingerprint: str,
    input_fingerprints: Tuple[Tuple[str, str], ...] = (),
    source_artifacts: Tuple[ArtifactRef, ...] = (),
    compile_options: Tuple[str, ...] = (),
    entry_points: Tuple[str, ...] = (),
    registry: Optional[JitSpecRegistry] = jit_spec_registry,
    kernel_id: Optional[str] = None,
    source: Optional[SourceRef] = None,
    output_path: Optional[str] = None,
) -> JitSpec:
    """Create a canonical spec and optionally publish it to a registry."""

    spec = JitSpec(
        name=name,
        domain=domain,
        generator_id=generator_id,
        generator_version=generator_version,
        target_soc=target_soc,
        environment_fingerprint=environment_fingerprint,
        specialization_fingerprint=specialization_fingerprint,
        input_fingerprints=input_fingerprints,
        source_artifacts=source_artifacts,
        compile_options=compile_options,
        entry_points=entry_points,
        kernel_id=kernel_id,
        source=source,
        output_path=output_path,
    )
    if registry is not None:
        registry.register(spec)
    return spec


__all__ = [
    "JIT_SPEC_SCHEMA_VERSION",
    "JitSpec",
    "JitSpecRegistry",
    "JitSpecStatus",
    "gen_jit_spec",
    "jit_spec_registry",
]
