"""Shared subprocess/build fixtures for every declaratively registered AOT operator."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .testing import local_vendor_names, select_aot_implementation, vendor_directory, vendor_search_path


def implementation_param(implementation):
    """Keep each kernel ID and its own hardware constraints visible to pytest."""
    import pytest
    from flashinfer.runtime import CompileMode

    descriptor = implementation.bind()
    socs = (descriptor.target_soc,) if descriptor.target_soc else descriptor.capability.soc_models
    marks = [pytest.mark.requires_soc(*socs)] if socs else []
    if descriptor.compile_mode in (CompileMode.AOT, CompileMode.JIT):
        marks.append(pytest.mark.real_compile)
    return pytest.param(implementation, id=descriptor.kernel_id, marks=marks)


def aot_params(op=None):
    from .implementation import implementations
    return [implementation_param(d) for d in implementations(backend="ascendc")
            if op is None or d.op == op]


def prepare_release(name, directory, *, kernel_id=None):
    import torch
    import torch_npu  # noqa: F401
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages
    from flashinfer.ascendc import load_contract
    from flashinfer.jit.discovery import AotPackageIndex
    from flashinfer.runtime import current_hardware

    assert torch.npu.is_available(), "missing NPU/runtime dependency is a failure"
    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    kernel_id = select_aot_implementation(name, kernel_id).kernel_id
    descriptor, _, spec = load_contract(kernel_id)
    assert current_hardware().soc_model == descriptor.target_soc, "requires the declared implementation target"
    release = os.environ.get("FLASHINFER_OPERATOR_TEST_RELEASE")
    if release is None:
        release = directory / "release"
        build_production_packages(kernel_ids=(kernel_id,), output_dir=release,
                                  cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", directory / "cache"))
    adapter = os.environ.get("FLASHINFER_OPERATOR_TEST_ADAPTER")
    if adapter is None:
        adapter = directory / "adapter"
        build_adapter(adapter)
    release = Path(release).resolve()
    match = AotPackageIndex(roots=(release,)).find(spec, descriptor=descriptor, target_soc=spec.target_soc)
    vendor = vendor_directory(match, directory)
    return dict(FLASHINFER_AOT_PATHS=str(release),
                ASCEND_CUSTOM_OPP_PATH=vendor_search_path([vendor], replace=local_vendor_names(name)),
                FLASHINFER_ACLNN_ADAPTER_PATH=str(Path(adapter).resolve()))


def execute(name, mode, directory, release, *, kernel_id=None, **overrides):
    environment = dict(os.environ, **release)
    if mode in ("missing", "corrupt", "jit"):
        # A cold worker must start before CANN registers any inherited vendors.
        environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    environment.update(FLASHINFER_NPU_JIT_DISABLE="0" if mode == "jit" else "1", FLASHINFER_NPU_BACKEND="auto")
    environment.update(overrides)
    command = [sys.executable, "-B", "-m", "flashinfer.operators.testing", name, mode]
    if kernel_id is not None:
        command.extend(["--kernel-id", kernel_id])
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
    (directory / (mode + ".log")).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    record = next(json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith('{"') and '"result": "PASS"' in line)
    assert record["operator"] == name and record["mode"] == mode
    if kernel_id is not None:
        assert record["kernel_id"] == kernel_id
    expected_calls = int(mode == "jit")
    assert record["compiler_calls"] == dict(compiler=expected_calls, toolchain=expected_calls, jit=expected_calls)
    return record


def damaged_release(name, directory, release, *, kernel_id=None):
    from flashinfer.ascendc import load_contract
    from flashinfer.jit.discovery import AotPackageIndex
    descriptor, _, spec = load_contract(select_aot_implementation(name, kernel_id).kernel_id)
    index = AotPackageIndex(roots=release["FLASHINFER_AOT_PATHS"].split(os.pathsep))
    source = index.find(spec, descriptor=descriptor, target_soc=spec.target_soc).directory
    return corrupt_package_copy(source, directory / "damaged")


def corrupt_package_copy(source, directory):
    """Flip one byte of the API library, leaving the original manifest and release intact."""
    damaged = directory / source.name
    shutil.copytree(source, damaged)
    package = json.loads((damaged / "op-package.json").read_text())
    library = damaged / package["api_library"]
    content = library.read_bytes()
    library.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    return damaged.parent
