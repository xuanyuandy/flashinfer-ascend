"""Prepare per-kernel acceptance releases through the framework AOT builder."""

import argparse
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from flashinfer._resources import data_root


ROOT = Path(__file__).resolve().parents[2]


def build_release(kernel_id, root):
    from flashinfer.aot.build import build_production_packages

    root = Path(root)
    return build_production_packages(
        output_dir=root / "release",
        cache_dir=root / "build-cache",
        kernel_ids=(kernel_id,),
    )[0]


@contextmanager
def installed_release(release):
    root = data_root(ROOT) / "op_packages"
    created = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / release.name
    destination.mkdir()  # Claim this fingerprint without replacing an installed package.
    try:
        shutil.copytree(release, destination, dirs_exist_ok=True)
        yield destination
    finally:
        shutil.rmtree(destination)
        if created:
            try:
                root.rmdir()
            except OSError as error:
                if error.errno != errno.ENOTEMPTY:
                    raise


def runtime_only_cann(directory):
    original = Path(os.environ["ASCEND_HOME_PATH"])
    runtime = directory / "runtime-only-cann"
    runtime.mkdir()
    for name in ("lib64", "share", "conf", "opp", "runtime"):
        (runtime / name).symlink_to(original / name, target_is_directory=True)
    (runtime / "compiler").mkdir()  # torch_npu checks for this directory at import.
    assert not (runtime / "bin/bisheng").exists()
    return {"ASCEND_HOME_PATH": str(runtime), "PATH": "/usr/bin:/bin"}


def run_release_scenario(module, scenario, *arguments, search_root=None, env=None,
                         python=sys.executable, timeout=240):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("FLASHINFER_AOT_PATHS", None)
    environment.pop("FLASHINFER_NPU_JIT_DISABLE", None)
    if search_root is not None:
        environment["FLASHINFER_AOT_PATHS"] = str(search_root)
    environment.update(env or {})
    command = [str(python), "-B", "-m", module, scenario, *map(str, arguments)]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"') and '"scenario"' in line)
    assert evidence["result"] == "PASS" and evidence["scenario"] == scenario
    print(json.dumps(evidence, sort_keys=True))
    return evidence


def scenario_main(run_scenario):
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("store_root", type=Path)
    args = parser.parse_args()
    run_scenario(args.scenario, args.store_root)
