"""Build an explicit release matrix: one named binary wheel per NPU family."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from flashinfer._wheel import selected_families


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc", action="append", required=True,
                        help="repeat Ascend910B / Ascend910C / Ascend950; B/C share one 910B wheel")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        targets = selected_families(args.soc)
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[2]
    if not (root / "setup.py").is_file():
        parser.error("wheel release builds require a source checkout")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        environment = dict(os.environ, FLASHINFER_AOT_SOCS=target)
        if not (environment.get("FLASHINFER_AOT_CONFIG") or environment.get("FLASHINFER_AOT_PREBUILT")):
            # A release matrix always builds binary wheels, even if the caller
            # previously opted out of native artifacts for a source-only build.
            environment["FLASHINFER_AOT_ALL"] = "1"
        result = subprocess.run([sys.executable, "-B", "setup.py", "bdist_wheel",
            "--dist-dir", str(args.output_dir.resolve()),
            "--bdist-dir", str(root / "build/wheel-release" / target)],
            cwd=root, env=environment)
        if result.returncode:
            raise SystemExit(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
