"""Isolate release checks that execute setup.py or replace provider modules.

These checks intentionally mutate process-wide build/provider state. Running
them in fresh interpreters keeps that state and its object lifetimes out of
the existing device/graph tests, including their strict memory-counter checks.
"""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("suite", ["aot_all", "wheel", "selftest"])
def test_release_contracts_in_fresh_process(suite):
    root = Path(__file__).resolve().parents[2]
    checks = root / "tests/release_checks" / (suite + "_checks.py")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", str(checks)],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
