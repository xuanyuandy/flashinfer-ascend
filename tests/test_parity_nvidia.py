import json
from pathlib import Path

import pytest

from flashinfer.parity import ParityEntry
from flashinfer.runtime import SchemaError


def test_nvidia_inventory_has_reasons_and_pinned_declarations():
    root = Path(__file__).parents[1] / "flashinfer/data"
    total = 0
    for name in ("api_parity.json", "attention_api_parity.json"):
        entries = json.loads((root / name).read_text())["entries"]
        assert len({e["upstream"] for e in entries}) == len(entries)
        for entry in entries:
            ParityEntry.from_dict(entry)
            if entry["semantic_status"] == "intentionally_different":
                assert entry["reason"].strip()
            if "upstream_source" in entry:
                source = entry["upstream_source"]
                assert source["commit"] == "2895577720741ddfedff118202c89584996b8875"
                assert source["path"].startswith("flashinfer/")
                assert source["line"] > 0
                assert len(source["sha256"]) == 64
                total += 1
    assert total >= 46


def test_intentional_difference_cannot_lose_its_reason():
    with pytest.raises(SchemaError, match="requires reason"):
        ParityEntry.from_dict(dict(
            upstream="flashinfer.comm.example", local="flashinfer.comm.example",
            upstream_kind="stable", semantic_status="intentionally_different",
            implementation_status="missing", priority="P3",
        ))
