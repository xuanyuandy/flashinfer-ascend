"""Device/build setup stays outside public-API numerical tests."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(scope="session")
def device_runtime():
    import torch
    import torch_npu

    if not torch.npu.is_available():
        pytest.fail("operator accuracy tests require an available NPU")
    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    return torch, torch_npu


@pytest.fixture
def npu_runtime(device_runtime):
    torch, torch_npu = device_runtime
    # TorchNPU exposes a setter here; its option API supplies the previous value.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    assert previous in (b"enable", b"disable"), "cannot preserve the NPU internal-format option"
    torch.npu.config.allow_internal_format = True
    try:
        yield torch, torch_npu
    finally:
        torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": previous.decode("ascii")})


def _prepare_local_aot(root):
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages
    from flashinfer.runtime import Backend, current_hardware, get_global_registry

    release = os.environ.get("FLASHINFER_OPERATOR_TEST_RELEASE")
    adapter = os.environ.get("FLASHINFER_OPERATOR_TEST_ADAPTER")
    if release is None:
        release = root / "release"
        target = current_hardware().soc_model
        kernel_ids = tuple(d.kernel_id for d in get_global_registry().descriptors()
                           if d.backend == Backend.ASCENDC and d.target_soc == target)
        assert kernel_ids, "the local SoC has no registered production AOT implementation"
        build_production_packages(kernel_ids=kernel_ids, output_dir=release,
                                  cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", root / "cache"))
    if adapter is None:
        adapter = root / "adapter"
        build_adapter(adapter)
    return {"FLASHINFER_AOT_PATHS": str(release), "FLASHINFER_ACLNN_ADAPTER_PATH": str(adapter)}


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    if session.testsfailed or session.config.option.collectonly or not any(
        "aot_runtime" in item.fixturenames or item.get_closest_marker("operator_aot") is not None
        for item in session.items
    ):
        return
    from flashinfer.operators.pytest_support import operator_test_vendors
    from flashinfer.operators.testing import vendor_search_path
    from flashinfer.runtime import current_hardware

    # CANN snapshots vendor search paths on its first operator, including CANN
    # operators from earlier tests in the same process. Prepare before test one.
    temporary = TemporaryDirectory(prefix="flashinfer-operator-tests-")
    session.config.add_cleanup(temporary.cleanup)
    root = Path(temporary.name)
    release = _prepare_local_aot(root)
    session.config._flashinfer_operator_release = release
    soc = current_hardware().soc_model
    vendors = [path for _, path in operator_test_vendors(release["FLASHINFER_AOT_PATHS"], root, target_soc=soc)]
    environment = pytest.MonkeyPatch()
    session.config.add_cleanup(environment.undo)
    environment.setenv("ASCEND_CUSTOM_OPP_PATH", vendor_search_path(vendors, replace={Path(p).name for p in vendors}))


@pytest.fixture(scope="session")
def local_aot_release(device_runtime, request, tmp_path_factory):
    release = getattr(request.config, "_flashinfer_operator_release", None)
    return release if release is not None else _prepare_local_aot(tmp_path_factory.mktemp("operator-accuracy-aot"))


@pytest.fixture
def aot_runtime(local_aot_release, monkeypatch):
    for name, value in local_aot_release.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")


@pytest.fixture(params=[
    "cann",
    pytest.param("ascendc", marks=[pytest.mark.real_compile, pytest.mark.requires_soc("Ascend950"),
                                  pytest.mark.operator_aot]),
    pytest.param("triton", marks=[pytest.mark.real_compile, pytest.mark.requires_soc("Ascend950")]),
])
def matmul_backend(request):
    if request.param == "ascendc":
        request.getfixturevalue("aot_runtime")
    return request.param
