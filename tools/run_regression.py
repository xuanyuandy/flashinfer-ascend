"""Run an explicit regression scope once and retain logs, counts and real exit status.

quick: all tests without real_compile; this is not a CPU-only or full result.
compile: real_compile tests only. full: the entire suite, including real_compile.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SCOPES = {"quick": "not real_compile", "compile": "real_compile", "full": None}


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - sum(counts[key] for key in ("failures", "errors", "skipped"))
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=SCOPES, default="quick")
    parser.add_argument("--log-dir", type=Path,
                        help="New directory for this run; an existing directory is rejected")
    parser.add_argument("paths", nargs="*", default=["tests"], help="Optional changed-area test paths")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    logs = (args.log_dir or root / "build/regression" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    logs.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *args.paths,
               "--junitxml=" + str(logs / "results.xml")]
    if SCOPES[args.profile]:
        command.extend(["-m", SCOPES[args.profile]])
    (logs / "command.json").write_text(json.dumps({"profile": args.profile, "command": command,
                                                  "cwd": str(root)}, indent=2) + "\n")
    print("profile=" + args.profile + "; log=" + str(logs / "pytest.log"), flush=True)
    if args.profile == "quick":
        print("Scope: real_compile is deselected; this run does not establish compiler or full coverage.", flush=True)
    started = time.monotonic()
    with (logs / "pytest.log").open("w") as output:
        result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
    (logs / "pytest.exit").write_text(str(result.returncode) + "\n")
    summary = {"profile": args.profile, "pytest_returncode": result.returncode,
               "elapsed_seconds": round(time.monotonic() - started, 3)}
    xml = logs / "results.xml"
    if xml.exists():
        summary.update(junit_counts(xml))
    output_text = (logs / "pytest.log").read_text()
    deselected = re.findall(r"(\d+) deselected", output_text)
    summary["deselected"] = int(deselected[-1]) if deselected else 0
    # A skipped device/optional-dependency test is not accepted as a green run.
    exit_code = (128 - result.returncode if result.returncode < 0 else result.returncode)
    if exit_code == 0 and (summary.get("skipped", 0) or summary.get("tests", 0) == 0):
        exit_code = 1
    summary["exit_code"] = exit_code
    (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (logs / "run.exit").write_text(str(exit_code) + "\n")
    print(output_text[-5000:], end="")
    print(json.dumps(summary, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
