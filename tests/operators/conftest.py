"""Device/build setup stays outside public-API numerical tests."""

import os

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


@pytest.fixture(scope="session")
def local_aot_release(device_runtime, tmp_path_factory):
    from flashinfer.aclnn.build import build_adapter
    from flashinfer.aot.build import build_production_packages
    from flashinfer.runtime import Backend, current_hardware, get_global_registry

    root = tmp_path_factory.mktemp("operator-accuracy-aot")
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


@pytest.fixture
def aot_runtime(local_aot_release, monkeypatch):
    for name, value in local_aot_release.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")
    monkeypatch.setenv("FLASHINFER_NPU_TUNING_DB", "")


@pytest.fixture(params=[
    "cann",
    pytest.param("ascendc", marks=[pytest.mark.real_compile, pytest.mark.requires_soc("Ascend950")]),
    pytest.param("triton", marks=[pytest.mark.real_compile, pytest.mark.requires_soc("Ascend950")]),
])
def matmul_backend(request):
    if request.param == "ascendc":
        request.getfixturevalue("aot_runtime")
    return request.param
