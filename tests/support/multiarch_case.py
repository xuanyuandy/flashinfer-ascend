"""Private vLLM-style multi-architecture operator; never in production inventory."""

import json
from pathlib import Path
import shutil

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.jit.tensor_adapter import package_tensor_runner

SOCS = ("Ascend910B", "Ascend910C", "Ascend950")
ROOT = Path(__file__).resolve().parents[2]


def infer(arguments, layouts):
    a, b = arguments["a"], arguments["b"]
    if tuple(a.shape) != (16, 16) or tuple(b.shape) != (16, 16):
        raise ValueError("multiarch probe requires 16x16 matrices")
    return Invocation((a, b), (OutputSpec((16, 16), "float32", a.device),), {}, None)


OPERATOR = OperatorDefinition(
    "multiarch_probe",
    "gemm",
    "test.multiarch_probe",
    schema="(Tensor a, Tensor b) -> Tensor",
    output_names=("out",),
    infer=infer,
    tensor_shapes=(("M", "K"), ("K", "N"), ("M", "N")),
)


def make_runner(module, spec, recipe):
    """One actual native runner factory, with no SoC/architecture branch."""
    return package_tensor_runner(module, recipe, spec=spec)


def project(directory):
    root = Path(directory)
    root.mkdir(parents=True)
    shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
    for name in ("CMakeLists.txt", "cmake"):
        source, target = ROOT / "csrc" / name, root / "csrc" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        (
            shutil.copytree(source, target)
            if source.is_dir()
            else shutil.copyfile(source, target)
        )
    native = root / "csrc/kernels/gemm/multiarch_probe"
    for name in (
        "op_host",
        "op_kernel/arch22",
        "op_kernel/arch35",
        "include/flashinfer_ascend",
    ):
        (native / name).mkdir(parents=True, exist_ok=True)
    base = ROOT / "csrc/kernels/gemm/matmul_probe"
    shutil.copyfile(
        base / "op_host/matmul_probe.cpp", native / "op_host/matmul_probe.cpp"
    )
    shutil.copyfile(
        base / "include/flashinfer_ascend/probe_tiling.h",
        native / "include/flashinfer_ascend/probe_tiling.h",
    )
    (native / "op_kernel/entry.cpp").write_text(
        """
#include "kernel_operator.h"
#include "flashinfer_ascend/probe_tiling.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/compute.h"
#else
#include "arch22/compute.h"
#endif
extern "C" __global__ __aicore__ void fi_multiarch_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    FI_PROBE_CHECK_TILING();
    Compute(reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
            reinterpret_cast<__gm__ float*>(out));
}
"""
    )
    (native / "op_kernel/arch22/compute.h").write_text(
        """
#pragma once
__aicore__ inline void Compute(__gm__ float* a, __gm__ float* b, __gm__ float* out) {
    for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) {
        float sum = 0;
        for (int k = 0; k < 16; ++k) sum += a[m * 16 + k] * b[k * 16 + n];
        out[m * 16 + n] = sum;
    }
}
"""
    )
    (native / "op_kernel/arch35/compute.h").write_text(
        """
#pragma once
__aicore__ inline void Compute(__gm__ float* a, __gm__ float* b, __gm__ float* out) {
    for (int index = 0; index < 256; ++index) {
        const int m = index / 16, n = index % 16;
        float sum = 0;
        for (int k = 0; k < 16; ++k) sum += a[m * 16 + k] * b[k * 16 + n];
        out[index] = sum;
    }
}
"""
    )
    logical = root / "flashinfer/operators/gemm/multiarch_probe"
    backend = root / "flashinfer/ascendc/gemm/multiarch_probe"
    logical.mkdir(parents=True)
    backend.mkdir(parents=True)
    (logical / "operator.py").write_text("from tests.support.multiarch_case import OPERATOR\n")
    (logical / "cases.py").write_text(
        "# Numeric golden is independently checked by the test driver.\n"
    )
    (backend / "implementation.py").write_text(
        """
from tests.support.multiarch_case import OPERATOR, SOCS, make_runner
from flashinfer.operators import Support, DimRange
from flashinfer.ascendc.definition import AscendCImplementation, BuildConfig, NativeBinding
from flashinfer.jit.recipe import TensorParameter

IMPLEMENTATIONS = AscendCImplementation.for_targets(
    operator=OPERATOR,
    support=Support(soc=SOCS, dtype="float32", layout="ND",
                    dimensions={key: DimRange(16, 16) for key in ("M", "N", "K")}),
    build=BuildConfig(device="op_kernel/entry.cpp", host=("op_host/matmul_probe.cpp",),
                      op_type="FiMultiarchProbe", tiling="FiMatmulProbeTiling",
                      includes=("include", "op_kernel"), task_type="aic_only",
                      sync_policy="compiler_default", workspace_bytes=0),
    binding=NativeBinding(tensors=(TensorParameter("a", "input", "float32", ("M", "K")),
        TensorParameter("b", "input", "float32", ("K", "N")),
        TensorParameter("c", "output", "float32", ("M", "N")))),
    runner_factory=make_runner,
)
"""
    )
    return root


def execute(root, release, adapter):
    """Exercise common dispatch/runtime/runner on actual hardware, without a build."""
    import os

    os.environ["FLASHINFER_ACLNN_ADAPTER_PATH"] = str(adapter)
    os.environ["FLASHINFER_NPU_JIT_DISABLE"] = "1"
    import torch
    import torch_npu
    from unittest.mock import patch
    from flashinfer._resources import data_root
    from flashinfer.runtime import (
        KernelRegistry,
        current_hardware,
        kernel_registry_scope,
    )
    from flashinfer.jit import AscendCToolchain
    from flashinfer.jit.discovery import AotPackageIndex
    from flashinfer.ascendc import _catalog
    from flashinfer.ascendc._runtime import bind_kernel
    from flashinfer.operators import runtime

    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    assert torch.npu.is_available(), "missing device is a failure"
    data = data_root(Path(root)) / "kernels"
    index = AotPackageIndex((Path(release),))
    # Use the production binder, including all target variants as package peers.
    # Only the private catalog/index location is injected for this subprocess.
    with patch.object(_catalog, "_DATA", data):
        _catalog._contracts.cache_clear()
        bound = [bind_kernel(key, make_runner) for key in _catalog._contracts()]
    registry = KernelRegistry(bound)
    registry.freeze()
    a = torch.randn(16, 16)
    b = torch.randn(16, 16)
    reference = a.double() @ b.double()
    a, b = a.npu(), b.npu()
    out = torch.empty_like(a)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime hit attempted compiler/toolchain discovery")

    with (
        kernel_registry_scope(registry),
        patch.object(AscendCToolchain, "discover", forbidden),
        patch.object(runtime, "definition", lambda _: OPERATOR),
        patch("flashinfer.jit.discovery.get_op_package_index", lambda: index),
    ):
        for output in (None, out):
            actual = runtime.execute(OPERATOR.op, OPERATOR.bind(a, b, out=output))
            if output is not None:
                assert actual is output
            torch.testing.assert_close(
                actual.cpu().double(), reference, atol=2e-5, rtol=2e-5
            )
    hardware = current_hardware()
    selected = [d for d in bound if d.runner.provenance is not None]
    assert len(selected) == 1 and selected[0].target_soc == hardware.soc_model
    assert selected[0].runner.provenance.source == "aot"
    print(
        json.dumps(
            dict(
                result="PASS",
                soc=hardware.soc_model,
                selected=selected[0].kernel_id,
                shared_runner=True,
                numerical_cases=2,
                runtime_compilations=0,
            )
        )
    )


if __name__ == "__main__":
    import sys

    execute(*sys.argv[1:])
