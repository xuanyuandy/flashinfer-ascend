"""Machine-readable FlashInfer API parity tracking."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Tuple, Union

from .runtime.schema import SchemaError


PARITY_SCHEMA_VERSION = 2
SUPPORTED_PARITY_SCHEMA_VERSIONS = {1, PARITY_SCHEMA_VERSION}
UPSTREAM_KINDS = {"stable", "experimental"}
SEMANTIC_STATUSES = {"unassessed", "exact", "compatible", "intentionally_different"}
IMPLEMENTATION_STATUSES = {
    "missing",
    "framework",
    "reference",
    "functional",
    "optimized",
}
CANN_COVERAGE_STATUSES = {"covered", "unavailable", "unverified", "not_applicable"}
CONSUMER_STATUSES = {"used", "not_used", "unverified"}
API_KINDS = {"upstream", "npu_extension", "compatibility_extension"}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ATTENTION_MODES = {
    "single_prefill",
    "single_decode",
    "batch_mixed_paged",
    "batch_prefill_paged",
    "batch_prefill_ragged",
    "batch_decode_paged",
}
PUBLIC_LIFECYCLES = {"one_shot", "plan_run"}
PROVIDER_ROUTING_KINDS = {"none", "ephemeral", "mode_bound"}
NPU_EXECUTION_STATUSES = {
    "none",
    "integration_required",
    "functional",
    "optimized",
}


@dataclass(frozen=True)
class ParityEntry:
    upstream: str
    local: str
    upstream_kind: str
    semantic_status: str
    implementation_status: str
    priority: str
    notes: str = ""
    reason: str = ""
    cann_coverage: str = "unverified"
    cann_operators: Tuple[str, ...] = ()
    local_status: str = ""
    consumers: Mapping[str, str] = field(default_factory=dict)
    api_kind: str = "upstream"
    upstream_source: Mapping[str, Any] = field(default_factory=dict)
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParityEntry":
        known = {
            "upstream", "local", "upstream_kind", "semantic_status",
            "implementation_status", "priority", "notes", "reason", "cann_coverage",
            "cann_operators", "local_status", "consumers", "api_kind", "upstream_source",
        }
        for name in ("upstream", "local", "upstream_kind", "semantic_status",
                     "implementation_status", "priority", "notes", "reason"):
            if name in value and not isinstance(value[name], str):
                raise SchemaError("%s must be a string" % name)
        operators = value.get("cann_operators", ())
        if not isinstance(operators, (list, tuple)) or not all(
            isinstance(name, str) and name.strip() for name in operators
        ):
            raise SchemaError("cann_operators must be a list of non-empty names")
        consumers = value.get("consumers", {})
        source = value.get("upstream_source", {})
        if not isinstance(consumers, Mapping) or not isinstance(source, Mapping):
            raise SchemaError("consumers and upstream_source must be objects")
        entry = cls(
            upstream=str(value["upstream"]),
            local=str(value["local"]),
            upstream_kind=str(value["upstream_kind"]),
            semantic_status=str(value["semantic_status"]),
            implementation_status=str(value["implementation_status"]),
            priority=str(value["priority"]),
            notes=str(value.get("notes", "")),
            reason=str(value.get("reason", "")),
            cann_coverage=str(value.get("cann_coverage", "unverified")),
            cann_operators=tuple(operators),
            local_status=str(value.get("local_status", "")),
            consumers=dict(consumers),
            api_kind=str(value.get("api_kind", "upstream")),
            upstream_source=dict(source),
            extra_fields={key: val for key, val in value.items() if key not in known},
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if not self.upstream.startswith("flashinfer."):
            raise SchemaError("invalid upstream symbol: %s" % self.upstream)
        if not self.local.startswith("flashinfer."):
            raise SchemaError("invalid local symbol: %s" % self.local)
        if self.upstream_kind not in UPSTREAM_KINDS:
            raise SchemaError("invalid upstream_kind: %s" % self.upstream_kind)
        if self.semantic_status not in SEMANTIC_STATUSES:
            raise SchemaError("invalid semantic_status: %s" % self.semantic_status)
        if self.semantic_status == "intentionally_different" and not self.reason.strip():
            raise SchemaError("intentionally_different entry requires reason")
        if self.implementation_status not in IMPLEMENTATION_STATUSES:
            raise SchemaError(
                "invalid implementation_status: %s" % self.implementation_status
            )
        if self.priority not in {"P0", "P1", "P2", "P3"}:
            raise SchemaError("invalid priority: %s" % self.priority)
        if self.cann_coverage not in CANN_COVERAGE_STATUSES:
            raise SchemaError("invalid cann_coverage: %s" % self.cann_coverage)
        if self.cann_coverage == "covered" and not self.cann_operators:
            raise SchemaError("covered CANN entry requires cann_operators")
        if self.api_kind not in API_KINDS:
            raise SchemaError("invalid api_kind: %s" % self.api_kind)
        if self.local_status and self.local_status != self.implementation_status:
            raise SchemaError("local_status must equal implementation_status")
        if set(self.consumers) - {"vllm", "sglang"} or any(
            status not in CONSUMER_STATUSES for status in self.consumers.values()
        ):
            raise SchemaError("invalid consumer usage status")
        if self.upstream_source:
            source = self.upstream_source
            if not _COMMIT.fullmatch(str(source.get("commit", ""))):
                raise SchemaError("upstream_source requires a full commit hash")
            if not _SHA256.fullmatch(str(source.get("sha256", ""))):
                raise SchemaError("upstream_source requires a source sha256")
            path = source.get("path", "")
            line = source.get("line")
            if (not isinstance(path, str) or not path.startswith("flashinfer/")
                    or ".." in Path(path).parts or type(line) is not int or line <= 0):
                raise SchemaError("upstream_source requires a source path and line")

    def validate_tracking(self, upstream_ref: str) -> None:
        if not self.local_status or set(self.consumers) != {"vllm", "sglang"}:
            raise SchemaError("entry lacks local_status or consumer tracking: " + self.upstream)
        if self.api_kind == "upstream":
            if self.upstream_source.get("commit") != upstream_ref:
                raise SchemaError("entry source is not pinned to ledger baseline: " + self.upstream)
        elif self.semantic_status != "intentionally_different" or not self.reason:
            raise SchemaError("local extension requires an intentional difference and reason")

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            self.extra_fields, upstream=self.upstream, local=self.local,
            upstream_kind=self.upstream_kind, semantic_status=self.semantic_status,
            implementation_status=self.implementation_status, priority=self.priority,
            notes=self.notes, reason=self.reason, cann_coverage=self.cann_coverage,
            cann_operators=list(self.cann_operators), local_status=self.local_status,
            consumers=dict(self.consumers), api_kind=self.api_kind,
            **({"upstream_source": dict(self.upstream_source)} if self.upstream_source else {}),
        )


@dataclass(frozen=True)
class AttentionParitySurface:
    """One user-facing Attention mode tracked independently of symbol counts."""

    surface_id: str
    local: str
    attention_mode: str
    public_lifecycle: str
    host_reference: bool
    provider_routing: str
    npu_execution: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionParitySurface":
        surface = cls(
            surface_id=str(value["surface_id"]),
            local=str(value["local"]),
            attention_mode=str(value["attention_mode"]),
            public_lifecycle=str(value["public_lifecycle"]),
            host_reference=value["host_reference"],
            provider_routing=str(value["provider_routing"]),
            npu_execution=str(value["npu_execution"]),
        )
        surface.validate()
        return surface

    def validate(self) -> None:
        if not self.surface_id or any(
            item not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for item in self.surface_id
        ):
            raise SchemaError("invalid Attention parity surface_id")
        if not self.local.startswith("flashinfer."):
            raise SchemaError("invalid Attention parity surface local symbol")
        if self.attention_mode not in ATTENTION_MODES:
            raise SchemaError("invalid Attention parity surface mode")
        if self.public_lifecycle not in PUBLIC_LIFECYCLES:
            raise SchemaError("invalid Attention parity public lifecycle")
        if not isinstance(self.host_reference, bool):
            raise SchemaError("Attention parity host_reference must be boolean")
        if self.provider_routing not in PROVIDER_ROUTING_KINDS:
            raise SchemaError("invalid Attention parity provider routing")
        if self.npu_execution not in NPU_EXECUTION_STATUSES:
            raise SchemaError("invalid Attention parity NPU execution status")
        if self.public_lifecycle == "one_shot" and self.provider_routing == "mode_bound":
            raise SchemaError("one-shot Attention cannot own a reusable mode-bound route")
        if self.public_lifecycle == "plan_run" and self.provider_routing == "ephemeral":
            raise SchemaError("plan/run Attention cannot use an ephemeral route")
        if self.npu_execution in {"functional", "optimized"} and (
            self.provider_routing == "none"
        ):
            raise SchemaError("callable NPU execution requires provider routing")


@dataclass(frozen=True)
class ParityManifest:
    schema_version: int
    upstream_project: str
    upstream_ref: str
    upstream_url: str
    scope: str
    scope_definition: str
    inventory_status: str
    generated_at: str
    entries: Tuple[ParityEntry, ...]
    attention_surfaces: Tuple[AttentionParitySurface, ...] = ()
    coverage_matrix: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParityManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            value: Dict[str, Any] = json.load(handle)
        schema_version = value.get("schema_version")
        if schema_version not in SUPPORTED_PARITY_SCHEMA_VERSIONS:
            raise SchemaError(
                "unsupported parity schema_version %r" % value.get("schema_version")
            )
        upstream = value.get("upstream", {})
        manifest = cls(
            schema_version=int(schema_version),
            upstream_project=str(upstream["project"]),
            upstream_ref=str(upstream["ref"]),
            upstream_url=str(upstream["url"]),
            scope=str(value.get("scope", "all")),
            scope_definition=str(value.get("scope_definition", "All tracked APIs")),
            inventory_status=str(value["inventory_status"]),
            generated_at=str(value["generated_at"]),
            entries=tuple(ParityEntry.from_dict(item) for item in value["entries"]),
            attention_surfaces=tuple(
                AttentionParitySurface.from_dict(item)
                for item in value.get("attention_surfaces", ())
            ),
            coverage_matrix=value.get("coverage_matrix", {}),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.inventory_status not in {"bootstrap", "complete"}:
            raise SchemaError("invalid inventory_status: %s" % self.inventory_status)
        upstream_symbols = tuple(entry.upstream for entry in self.entries)
        local_symbols = tuple(entry.local for entry in self.entries)
        if len(upstream_symbols) != len(set(upstream_symbols)):
            raise SchemaError("parity manifest contains duplicate upstream symbols")
        if len(local_symbols) != len(set(local_symbols)):
            raise SchemaError("parity manifest contains duplicate local symbols")
        if not isinstance(self.coverage_matrix, Mapping):
            raise SchemaError("coverage_matrix must be an object")
        if self.coverage_matrix and self.coverage_matrix.get("status") not in {
            "missing_source", "partial", "verified",
        }:
            raise SchemaError("invalid coverage_matrix status")
        surface_ids = tuple(item.surface_id for item in self.attention_surfaces)
        surface_locals = tuple(item.local for item in self.attention_surfaces)
        surface_modes = tuple(item.attention_mode for item in self.attention_surfaces)
        if len(surface_ids) != len(set(surface_ids)):
            raise SchemaError("parity manifest contains duplicate Attention surfaces")
        if len(surface_locals) != len(set(surface_locals)):
            raise SchemaError("parity manifest maps multiple surfaces to one local symbol")
        if len(surface_modes) != len(set(surface_modes)):
            raise SchemaError("parity manifest contains duplicate Attention modes")
        entry_statuses = {
            entry.local: entry.implementation_status for entry in self.entries
        }
        for surface in self.attention_surfaces:
            status = entry_statuses.get(surface.local)
            if status is None:
                raise SchemaError(
                    "Attention parity surface local symbol is not in the inventory"
                )
            if status == "missing":
                raise SchemaError("represented Attention surface cannot be missing")
            if surface.host_reference and status not in {
                "reference",
                "functional",
                "optimized",
            }:
                raise SchemaError(
                    "Attention surface Host reference is not reflected by its symbol"
                )
            if surface.npu_execution in {"functional", "optimized"} and status not in {
                "functional",
                "optimized",
            }:
                raise SchemaError(
                    "Attention surface NPU execution is not reflected by its symbol"
                )

    @property
    def is_complete(self) -> bool:
        return (self.inventory_status == "complete"
                and self.coverage_matrix.get("status") == "verified" and all(
            entry.semantic_status == "intentionally_different" or (
                entry.implementation_status in {"functional", "optimized"}
                and entry.semantic_status != "unassessed"
            )
            for entry in self.entries
            if entry.upstream_kind == "stable" and entry.api_kind == "upstream"
            and not entry.extra_fields.get("upstream_alias_of")
        ))

    def validate_tracking(self) -> None:
        """Release-independent integrity gate, also runnable on a CPU-only host."""
        if not _COMMIT.fullmatch(self.upstream_ref):
            raise SchemaError("ledger baseline must be a full upstream commit hash")
        if not self.coverage_matrix:
            raise SchemaError("ledger lacks coverage_matrix provenance")
        if self.inventory_status == "complete" and self.coverage_matrix["status"] != "verified":
            raise SchemaError("complete inventory requires a verified coverage matrix")
        for entry in self.entries:
            entry.validate_tracking(self.upstream_ref)

    @property
    def pending_gaps(self) -> Tuple[ParityEntry, ...]:
        return tuple(
            entry for entry in self.entries
            if entry.api_kind == "upstream"
            and not entry.extra_fields.get("upstream_alias_of")
            and entry.semantic_status != "intentionally_different"
            and (entry.implementation_status not in {"functional", "optimized"}
                 or entry.semantic_status == "unassessed")
        )

    def counts(self) -> Counter:
        return Counter(entry.implementation_status for entry in self.entries)

    def attention_surface_counts(self) -> Counter:
        counts = Counter()
        for surface in self.attention_surfaces:
            counts["host_reference"] += int(surface.host_reference)
            counts["provider_routed"] += int(surface.provider_routing != "none")
            counts["npu_callable"] += int(
                surface.npu_execution in {"functional", "optimized"}
            )
        return counts

    def report(self) -> str:
        counts = self.counts()
        lines = [
            "FlashInfer API parity",
            "  upstream: %s (%s)" % (self.upstream_project, self.upstream_ref),
            "  scope: %s" % self.scope,
            "  inventory: %s" % self.inventory_status,
            "  symbols: %d" % len(self.entries),
            "  upstream-symbols: %d" % sum(
                entry.api_kind == "upstream" and not entry.extra_fields.get("upstream_alias_of")
                for entry in self.entries
            ),
            "  coverage-matrix: %s" % self.coverage_matrix.get("status", "unverified"),
            "  intentionally-different: %d" % sum(
                entry.semantic_status == "intentionally_different" for entry in self.entries
            ),
            "  local-extensions: %d" % sum(entry.api_kind != "upstream" for entry in self.entries),
            "  pending-gaps: %d" % len(self.pending_gaps),
        ]
        for status in (
            "optimized",
            "functional",
            "reference",
            "framework",
            "missing",
        ):
            lines.append("  %-10s %d" % (status + ":", counts[status]))
        if self.attention_surfaces:
            surface_counts = self.attention_surface_counts()
            lines.extend(
                (
                    "  attention-surfaces: %d" % len(self.attention_surfaces),
                    "  host-reference-surfaces: %d"
                    % surface_counts["host_reference"],
                    "  provider-routed-surfaces: %d"
                    % surface_counts["provider_routed"],
                    "  npu-callable-surfaces: %d" % surface_counts["npu_callable"],
                )
            )
        lines.append("  release-compatible: %s" % ("yes" if self.is_complete else "no"))
        if self.coverage_matrix.get("status") == "missing_source":
            lines.append("  note: original 147-row coverage matrix is unavailable; CANN/consumer coverage remains unverified where evidence is absent.")
        return "\n".join(lines)


def packaged_manifest_path(scope: str = "all") -> Path:
    filename = {
        "all": "api_parity.json",
        "attention": "attention_api_parity.json",
    }.get(scope)
    if filename is None:
        raise ValueError("unknown parity scope: %s" % scope)
    return Path(__file__).resolve().parent / "data" / filename


def load_packaged_manifest(scope: str = "all") -> ParityManifest:
    return ParityManifest.load(packaged_manifest_path(scope))


def validate_packaged_manifests() -> Tuple[ParityManifest, ...]:
    manifests = tuple(load_packaged_manifest(scope) for scope in ("all", "attention"))
    if len({manifest.upstream_ref for manifest in manifests}) != 1:
        raise SchemaError("packaged parity ledgers use different upstream commits")
    for manifest in manifests:
        manifest.validate_tracking()
    general = {entry.upstream: entry for entry in manifests[0].entries}
    for entry in manifests[1].entries:
        other = general.get(entry.upstream)
        if other is None:
            raise SchemaError("Attention symbol absent from general ledger: " + entry.upstream)
        if (entry.semantic_status, entry.implementation_status, entry.api_kind) != (
            other.semantic_status, other.implementation_status, other.api_kind,
        ):
            raise SchemaError("inconsistent Attention status between ledgers: " + entry.upstream)
    return manifests
