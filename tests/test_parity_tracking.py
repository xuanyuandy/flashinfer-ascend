from dataclasses import replace
import json

import pytest

from flashinfer.cli import main
from flashinfer.parity import (
    ParityEntry, load_packaged_manifest, validate_packaged_manifests,
)
from flashinfer.runtime import SchemaError


REF = "2895577720741ddfedff118202c89584996b8875"


def test_both_ledgers_keep_the_pinned_source_and_tracking_metadata():
    manifests = validate_packaged_manifests()
    assert {manifest.upstream_ref for manifest in manifests} == {REF}
    assert all(not manifest.is_complete for manifest in manifests)
    for manifest in manifests:
        assert manifest.coverage_matrix["status"] == "missing_source"
        for entry in manifest.entries:
            assert entry.local_status == entry.implementation_status
            assert entry.consumers == {"vllm": "unverified", "sglang": "unverified"}
            if entry.api_kind == "upstream":
                assert entry.upstream_source["commit"] == REF
            else:
                assert entry.semantic_status == "intentionally_different"
                assert entry.reason


def test_parameter_semantics_and_extra_source_evidence_survive_round_trip():
    manifest = load_packaged_manifest("all")
    entry = next(e for e in manifest.entries if e.upstream == "flashinfer.norm.rmsnorm")
    raw = entry.to_dict()
    assert raw["parameters"]["enable_pdl"]["semantic_status"] == "compatible"
    assert raw["cann_operators"] == ["npu_rms_norm"]
    assert raw["upstream_source"]["declaration"] == "flashinfer.norm.rmsnorm"
    raw["future_metadata"] = {"source": "reviewed-evidence", "versions": [1, 2]}
    assert ParityEntry.from_dict(json.loads(json.dumps(raw))).to_dict() == raw

    # Refreshing the declaration inventory must also keep a subsequently supplied
    # coverage-matrix row, even when the local interface is not implemented yet.
    import ast
    from work_items.scripts.update_parity_inventory import annotate

    class Source:
        def resolve(self, symbol):
            module, name = symbol.rsplit(".", 1)
            text = "def %s(): pass\n" % name
            return module.replace(".", "/") + ".py", text, ast.parse(text).body[0], symbol

    rope = next(e for e in manifest.entries if e.upstream == "flashinfer.rope.apply_rope").to_dict()
    rope.update(cann_coverage="covered", cann_operators=["npu_apply_rotary_pos_emb"],
                coverage_evidence={"matrix": "provided-coverage-table", "row": 3},
                consumers={"vllm": "used", "sglang": "unverified"})
    refreshed = annotate(dict(rope), Source())
    for name in ("cann_coverage", "cann_operators", "coverage_evidence", "consumers"):
        assert refreshed[name] == rope[name]
    # Earlier task commits used these draft values; migration must work from
    # those commits as well as when replayed against an already refreshed file.
    for status in ("verified", "verified_composite"):
        activation = next(e for e in manifest.entries
                          if e.upstream == "flashinfer.activation.silu_and_mul").to_dict()
        activation["cann_coverage"] = status
        activation.pop("coverage_evidence", None)
        migrated = ParityEntry.from_dict(annotate(activation, Source()))
        assert migrated.cann_coverage == "covered"
        assert migrated.cann_operators == ("npu_swiglu",)


@pytest.mark.parametrize("patch, message", [
    ({"cann_coverage": "guessed"}, "cann_coverage"),
    ({"cann_coverage": "covered", "cann_operators": []}, "requires cann_operators"),
    ({"cann_operators": "npu_rms_norm"}, "list"),
    ({"consumers": {"vllm": "maybe"}}, "consumer"),
    ({"local_status": "missing"}, "local_status"),
    ({"upstream_source": {"commit": "main"}}, "full commit"),
    ({"semantic_status": "intentionally_different", "reason": None}, "reason must be a string"),
])
def test_tracking_metadata_cannot_silently_overstate_coverage(patch, message):
    raw = next(e for e in load_packaged_manifest("all").entries
               if e.upstream == "flashinfer.norm.rmsnorm").to_dict()
    raw.update(patch)
    with pytest.raises(SchemaError, match=message):
        ParityEntry.from_dict(raw)


def test_tracking_rejects_moving_baselines_and_cross_commit_sources():
    manifest = load_packaged_manifest("all")
    with pytest.raises(SchemaError, match="full upstream commit"):
        replace(manifest, upstream_ref="main").validate_tracking()
    with pytest.raises(SchemaError, match="not pinned"):
        replace(manifest, upstream_ref="a" * 40).validate_tracking()
    with pytest.raises(SchemaError, match="verified coverage matrix"):
        replace(manifest, inventory_status="complete").validate_tracking()


def test_intentional_vendor_differences_are_excluded_from_pending_gaps():
    manifest = load_packaged_manifest("all")
    vendor = tuple(e for e in manifest.entries if ".comm." in e.upstream)
    assert len(vendor) >= 46
    assert all(e.semantic_status == "intentionally_different" for e in vendor)
    assert not set(e.upstream for e in vendor) & set(e.upstream for e in manifest.pending_gaps)
    # A deliberately unsupported vendor ABI can be resolved without a fake NPU implementation.
    resolved = replace(manifest, inventory_status="complete", entries=vendor,
                       coverage_matrix={"status": "verified"})
    assert resolved.is_complete
    assert not resolved.pending_gaps


def test_cli_distinguishes_ledger_integrity_from_completed_coverage(capsys):
    assert main(["parity-check"]) == 0
    output = capsys.readouterr().out
    assert "Parity ledger integrity: passed" in output
    assert "missing_source" in output
    assert "release-compatible: no" in output
    assert main(["parity-check", "--require-complete"]) == 1


def test_cli_reports_invalid_ledger_as_an_actionable_error(monkeypatch, capsys):
    def broken():
        raise SchemaError("entry source is not pinned to ledger baseline")
    monkeypatch.setattr("flashinfer.cli.validate_packaged_manifests", broken)
    assert main(["parity-check"]) == 2
    assert "not pinned" in capsys.readouterr().err


def test_conflicting_shared_attention_statuses_are_rejected(monkeypatch):
    general = load_packaged_manifest("all")
    attention = load_packaged_manifest("attention")
    changed = replace(attention.entries[0], implementation_status="missing", local_status="missing")
    attention = replace(attention, entries=(changed, *attention.entries[1:]))
    monkeypatch.setattr("flashinfer.parity.load_packaged_manifest",
                        lambda scope: general if scope == "all" else attention)
    with pytest.raises(SchemaError, match="inconsistent Attention status"):
        validate_packaged_manifests()
