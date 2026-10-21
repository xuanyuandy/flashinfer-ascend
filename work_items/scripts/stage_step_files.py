"""Stage only a recorded step's delta, preserving preexisting working edits."""

import difflib
import gzip
import json
from pathlib import Path
import subprocess
import sys

step, *names = sys.argv[1:]
with gzip.open(f"work_items/reports/execution/steps/{step}.before.json.gz", "rt") as stream:
    before = json.load(stream)
patch = "".join(line for name in names for line in difflib.unified_diff(
    before.get(name, "").splitlines(keepends=True),
    Path(name).read_text().splitlines(keepends=True),
    fromfile="a/" + name, tofile="b/" + name,
))
subprocess.run(["git", "apply", "--cached", "--whitespace=nowarn", "-"], input=patch, text=True, check=True)
