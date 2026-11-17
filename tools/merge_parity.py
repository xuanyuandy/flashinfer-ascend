"""Merge reviewed API compatibility entries into a parity ledger."""

import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("entries", type=Path)
parser.add_argument("--ledger", default="tools/metadata/api_parity.json")
parser.add_argument("--stage-only", action="store_true")
args = parser.parse_args()
path = Path(args.ledger)
raw = (subprocess.check_output(["git", "show", "HEAD:" + str(path)], text=True)
       if args.stage_only else path.read_text())
manifest = json.loads(raw)
entries = {e["upstream"]: e for e in manifest["entries"]}
for entry in json.loads(args.entries.read_text()):
    if "_patch" in entry:
        entries[entry["upstream"]].update(entry["_patch"])
        for old, new in entry.get("_notes_replace", ()):
            current = entries[entry["upstream"]]
            current["notes"] = current.get("notes", "").replace(old, new)
    else:
        entries[entry["upstream"]] = entry
manifest["entries"] = list(entries.values())
data = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
if args.stage_only:
    blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=data, text=True).strip()
    subprocess.run(["git", "update-index", "--cacheinfo", "100644," + blob + "," + str(path)], check=True)
else:
    path.write_text(data)
