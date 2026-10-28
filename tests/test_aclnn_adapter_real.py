"""Native Torch/ACLNN public interfaces retain verified AOT execution."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def aclnn_release(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages

    assert torch.npu.is_available() and torch.npu.device_count() == 8
    root = tmp_path_factory.mktemp("native-aclnn")
    build_adapter(root / "adapter")
    build_production_packages(output_dir=root / "release", cache_dir=root / "cache")
    return root


@pytest.mark.real_compile
@pytest.mark.parametrize("case", ["matmul", "attention"])
def test_native_aclnn_public_interfaces(aclnn_release, case):
    environment = dict(os.environ, FLASHINFER_ACLNN_ADAPTER_PATH=str(aclnn_release / "adapter"),
                       FLASHINFER_AOT_PATHS=str(aclnn_release / "release"),
                       FLASHINFER_NPU_JIT_DISABLE="1")
    environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    result = subprocess.run([sys.executable, "-B", "-m", "tests.aclnn_acceptance", case],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=180)
    print(result.stdout, end="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"result": "PASS"' in result.stdout
    assert '"adapter": "C++/OpCommand"' in result.stdout
