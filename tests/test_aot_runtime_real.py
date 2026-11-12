"""P0-2 real CANN acceptance: public API -> dispatch -> AOT/JIT -> NPU."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from flashinfer._resources import data_root

import pytest

from tests.aot_runtime_probe import ROOT, build_release


@pytest.fixture(scope="module")
def real_release(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires a real NPU; skips are not P0-2 acceptance")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/absent"))
    if not (cann / "bin/bisheng").is_file():
        pytest.skip("requires bisheng to prepare the real AOT release")
    return build_release(tmp_path_factory.mktemp("p0-2-aot"))


@contextmanager
def installed_release(release):
    """Install only this fixture in the real default wheel discovery layout."""
    root = data_root(ROOT) / "op_packages"
    created_root = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / release.name
    assert not destination.exists(), "never overwrite a pre-existing release"
    shutil.copytree(release, destination)
    try:
        yield destination
    finally:
        shutil.rmtree(destination)
        if created_root:
            root.rmdir()


def execute(scenario, release, tmp_path, *, search_root=None, second_release=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("FLASHINFER_AOT_PATHS", None)
    environment.pop("FLASHINFER_NPU_JIT_DISABLE", None)
    if search_root is not None:
        environment["FLASHINFER_AOT_PATHS"] = str(search_root)
    command = [sys.executable, "-B", "-m", "tests.aot_runtime_probe", scenario,
               str(release), str(tmp_path / "cold-runtime-store")]
    if second_release is not None:
        command.append(str(second_release))
    result = subprocess.run(command, cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = next(json.loads(line) for line in result.stdout.splitlines()
                    if line.startswith('{"') and '"scenario"' in line)
    assert evidence["result"] == "PASS" and evidence["scenario"] == scenario
    print(json.dumps(evidence, sort_keys=True))
    return evidence


@pytest.mark.real_compile
def test_aot_hit_with_jit_disabled_default_packaged_path(real_release, tmp_path):
    with installed_release(real_release):
        evidence = execute("hit_disabled", real_release, tmp_path)
    assert evidence["source"] == "aot" and evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_missing_allows_real_cold_jit_compile(real_release, tmp_path):
    evidence = execute("missing_enabled", real_release, tmp_path)
    assert evidence["source"] == "jit" and evidence["compiler_calls"] == 1


@pytest.mark.real_compile
@pytest.mark.parametrize("scenario", ["missing_disabled", "missing_cache_only"])
def test_missing_forbidden_jit_reports_explicit_error(real_release, tmp_path, scenario):
    evidence = execute(scenario, real_release, tmp_path)
    assert evidence["category"] == "missing" and not evidence["fallback"]


@pytest.mark.real_compile
def test_corrupt_digest_is_fatal_even_with_jit_enabled(real_release, tmp_path):
    root = tmp_path / "damaged-releases"
    damaged = root / real_release.name
    shutil.copytree(real_release, damaged)
    package = json.loads((damaged / "op-package.json").read_text())
    library = damaged / package["api_library"]
    payload = library.read_bytes()
    library.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    evidence = execute("corrupt", real_release, tmp_path, search_root=root)
    assert evidence["category"] == "corrupt" and evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_incompatible_soc_has_candidate_reason(real_release, tmp_path):
    other = build_release(tmp_path / "other-soc", target="ascend910b1")
    evidence = execute("incompatible", real_release, tmp_path, search_root=other.parent)
    assert evidence["category"] == "missing" and evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_missing_aclnn_symbol_is_load_error_without_jit(real_release, tmp_path):
    from flashinfer.jit import DiskJitBuildStore, JitSpec
    from flashinfer.jit.cache import verify_build_directory

    record = verify_build_directory(real_release)
    spec = JitSpec.from_dict(json.loads((real_release / "manifest.json").read_text())["spec"])
    package = json.loads((real_release / "op-package.json").read_text())
    store = DiskJitBuildStore(tmp_path / "symbol-fixture")
    with store.staging(spec) as staging:
        shutil.copytree(real_release, staging, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("manifest.json", "build.log"))
        library = staging / package["api_library"]
        symbol = package["recipe"]["host_entry"].encode()
        payload = library.read_bytes()
        assert symbol in payload
        library.write_bytes(payload.replace(symbol, b"x" + symbol[1:]))
        # Publish intentional missing-symbol bytes with a valid store inventory:
        # this is a loader ABI rejection, distinct from the corrupt-digest case.
        replacement = store.publish(spec, staging, record.artifact)
    directory = store.directory(replacement)
    verify_build_directory(directory, spec=spec, record=replacement)
    evidence = execute("load_error", real_release, tmp_path, search_root=directory)
    assert evidence["category"] == "load error" and evidence["compiler_calls"] == 0


@pytest.mark.real_compile
def test_second_package_identity_refused_without_jit(real_release, tmp_path):
    from flashinfer.jit import DiskJitBuildStore, JitSpec
    from flashinfer.jit.cache import verify_build_directory

    record = verify_build_directory(real_release)
    spec = JitSpec.from_dict(json.loads((real_release / "manifest.json").read_text())["spec"])
    store = DiskJitBuildStore(tmp_path / "second-identity")
    with store.staging(spec) as staging:
        shutil.copytree(real_release, staging, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("manifest.json", "build.log"))
        # Same executable/ABI, distinct valid immutable build-record identity.
        receipt = staging / "ascendc-recipe.json"
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        second = store.publish(spec, staging, record.artifact)
    assert second.fingerprint != record.fingerprint
    evidence = execute("identity", real_release, tmp_path, search_root=real_release,
                       second_release=store.directory(second))
    assert evidence["category"] == "load error" and evidence["first_numerical"]
