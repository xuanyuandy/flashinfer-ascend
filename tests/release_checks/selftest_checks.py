"""Installed selftest must fail closed without sources, devices, or artifacts."""

from dataclasses import asdict
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer import selftest
from flashinfer.cli import main


def test_missing_local_artifacts_cannot_use_extra_aot_roots(monkeypatch, tmp_path):
    descriptor = selftest._production_descriptors()[0]
    bundled = tmp_path / "installed" / "data" / "op_packages"
    external = tmp_path / "external" / ("a" * 64)
    external.mkdir(parents=True)
    # Searching this external root would report corruption, so a local miss
    # establishes that FLASHINFER_AOT_PATHS does not expand selftest's search.
    (external / "manifest.json").write_text("invalid JSON", encoding="utf-8")
    monkeypatch.setenv("FLASHINFER_AOT_PATHS", str(external.parent))
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    monkeypatch.setattr(selftest, "_package_root", lambda: bundled)
    monkeypatch.setattr(selftest, "_load_dependencies", lambda: object())
    result = selftest._check_kernel(descriptor)
    assert result.status == "SKIP"
    assert "AOT artifact not bundled" in result.reason
    assert "compilation=disabled" in result.reason
    assert str(bundled) in result.reason
    assert str(external) not in result.reason


def test_corrupt_bundled_artifact_is_failure(monkeypatch, tmp_path):
    descriptor = selftest._production_descriptors()[0]
    release = tmp_path / ("b" * 64)
    release.mkdir()
    (release / "manifest.json").write_text("invalid JSON", encoding="utf-8")
    monkeypatch.setattr(selftest, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(selftest, "_load_dependencies", lambda: object())
    result = selftest._check_kernel(descriptor)
    assert result.status == "FAIL"
    assert "corrupt AOT op-package" in result.reason


def test_no_device_is_explicit_skip(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        npu=SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    result = selftest._check_kernel(selftest._production_descriptors()[0])
    assert result.status == "SKIP"
    assert result.reason == "no NPU device is available"


def test_missing_runtime_dependency_is_explicit_skip(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(selftest._SelftestSkip, match="runtime dependency missing"):
        selftest._load_dependencies()


def test_new_production_op_without_numerical_case_fails():
    result = selftest._check_kernel(SimpleNamespace(
        op="new.production.op", kernel_id="new/production/kernel",
    ))
    assert result.status == "FAIL"
    assert "no numerical selftest registered" in result.reason


@pytest.mark.parametrize("statuses, exit_code", [
    (("PASS", "PASS"), 0), (("PASS", "SKIP"), 1), (("FAIL", "PASS"), 1),
])
def test_cli_visits_registry_and_never_counts_skip_as_pass(
    monkeypatch, capsys, statuses, exit_code,
):
    descriptors = selftest._production_descriptors()
    seen = []

    def run(kernel_id):
        seen.append(kernel_id)
        return selftest.SelftestResult(kernel_id, statuses[len(seen) - 1], "reason")

    monkeypatch.setattr(selftest, "_run_kernel_process", run)
    assert main(["selftest"]) == exit_code
    assert seen == [item.kernel_id for item in descriptors]
    output = capsys.readouterr().out
    assert "Summary: %d PASS, %d FAIL, %d SKIP (total=%d)" % (
        statuses.count("PASS"), statuses.count("FAIL"), statuses.count("SKIP"),
        len(descriptors),
    ) in output


def test_worker_uses_current_interpreter_and_disables_external_roots(monkeypatch):
    monkeypatch.setenv("FLASHINFER_AOT_PATHS", "/untrusted/external/aot")
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")
    expected = selftest.SelftestResult("production/kernel", "PASS", "numerical match")

    def run(command, **kwargs):
        assert command == [sys.executable, "-B", "-m", "flashinfer.selftest",
                           "--kernel", expected.kernel_id]
        assert kwargs["env"]["FLASHINFER_NPU_JIT_DISABLE"] == "1"
        assert kwargs["env"]["FLASHINFER_AOT_PATHS"] == ""
        assert "cwd" not in kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout="runtime log\n" + selftest._RESULT_PREFIX
            + json.dumps(asdict(expected)) + "\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert selftest._run_kernel_process(expected.kernel_id) == expected


def test_worker_crash_cannot_report_pass(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], -11, stdout="", stderr="crash"))
    result = selftest._run_kernel_process("production/kernel")
    assert result.status == "FAIL"
    assert "exit=-11" in result.reason and "crash" in result.reason


def test_empty_registry_is_not_success(monkeypatch, capsys):
    monkeypatch.setattr(selftest, "_production_descriptors", lambda: ())
    assert selftest.run_selftest() == 1
    assert "production registry is empty" in capsys.readouterr().out
