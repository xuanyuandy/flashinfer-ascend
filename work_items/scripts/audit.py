"""Capture source diffs and command output for the authorized task execution."""

import datetime
import difflib
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_items/reports/execution"


def snapshot():
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, text=True,
    ).splitlines()
    result = {}
    for name in sorted(set(paths)):
        if name.startswith("work_items/reports/execution/"):
            continue
        path = ROOT / name
        if path.is_file():
            try:
                result[name] = path.read_text()
            except UnicodeDecodeError:
                continue
    return result


def event(task, action, **extra):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(
            time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            task=task, action=action, **extra,
        ), ensure_ascii=False) + "\n")


def main():
    action, step, *args = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / "steps" / step
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if action == "begin":
        with gzip.open(str(prefix) + ".before.json.gz", "wt") as stream:
            json.dump(snapshot(), stream, ensure_ascii=False)
        event(step, "begin", description=" ".join(args))
    elif action == "end":
        with gzip.open(str(prefix) + ".before.json.gz", "rt") as stream:
            before = json.load(stream)
        after = snapshot()
        diff = "".join(
            line
            for name in sorted(before.keys() | after.keys())
            for line in difflib.unified_diff(
                before.get(name, "").splitlines(keepends=True),
                after.get(name, "").splitlines(keepends=True),
                fromfile="a/" + name, tofile="b/" + name,
            )
        )
        Path(str(prefix) + ".patch").write_text(diff)
        event(step, "end", patch=str(prefix.relative_to(ROOT)) + ".patch",
              changed_files=[n for n in sorted(before.keys() | after.keys())
                             if before.get(n) != after.get(n)])
    elif action == "run":
        result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        Path(str(prefix) + ".log").write_text(result.stdout)
        event(step, "command", command=args, exit_code=result.returncode,
              log=str(prefix.relative_to(ROOT)) + ".log")
        print(result.stdout[-6000:])
        raise SystemExit(result.returncode)
    else:
        raise SystemExit("expected begin, end or run")


if __name__ == "__main__":
    main()
