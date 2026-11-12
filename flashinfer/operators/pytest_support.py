"""Shared subprocess/build fixtures for every declaratively registered AOT operator."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .catalog import definition


def prepare_release(name, directory):
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages
    from flashinfer.runtime import current_hardware

    assert torch.npu.is_available(), "missing NPU/runtime dependency is a failure"
    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    operator = definition(name)
    from .implementation import implementations
    choices = [d for d in implementations(backend="ascendc") if d.op == operator.op]
    assert len(choices) == 1, "acceptance requires an explicit AOT implementation"
    kernel_id = choices[0].kernel_id
    from flashinfer.ascendc import load_contract
    assert current_hardware().soc_model == load_contract(kernel_id)[0].target_soc, "requires the declared implementation target"
    release = os.environ.get("FLASHINFER_OPERATOR_TEST_RELEASE")
    if release is None:
        release = directory / "release"
        build_production_packages(kernel_ids=(kernel_id,), output_dir=release,
                                  cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", directory / "cache"))
    adapter = os.environ.get("FLASHINFER_OPERATOR_TEST_ADAPTER")
    if adapter is None:
        adapter = directory / "adapter"
        build_adapter(adapter)
    return dict(FLASHINFER_AOT_PATHS=str(Path(release).resolve()),
                FLASHINFER_ACLNN_ADAPTER_PATH=str(Path(adapter).resolve()))


def execute(name, mode, directory, release, **overrides):
    environment = dict(os.environ, **release)
    environment.update(FLASHINFER_NPU_JIT_DISABLE="1", FLASHINFER_NPU_BACKEND="auto")
    environment.update(overrides)
    command = [sys.executable, "-B", "-m", "flashinfer.operators.testing", name, mode]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
    (directory / (mode + ".log")).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    record = next(json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith('{"') and '"result": "PASS"' in line)
    assert record["compiler_calls"] == dict(compiler=0, toolchain=0, jit=0)
    return record


def damaged_release(name, directory, release):
    from flashinfer.ascendc import load_contract
    from .implementation import implementations
    choices = [d for d in implementations(backend="ascendc") if d.op == definition(name).op]
    assert len(choices) == 1
    spec = load_contract(choices[0].kernel_id)[2]
    source = Path(release["FLASHINFER_AOT_PATHS"]) / spec.fingerprint
    damaged = directory / "damaged" / spec.fingerprint
    shutil.copytree(source, damaged)
    package = json.loads((damaged / "op-package.json").read_text())
    library = damaged / package["api_library"]
    content = library.read_bytes()
    library.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    return damaged.parent
