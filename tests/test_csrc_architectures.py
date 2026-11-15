"""Actual csrc architecture branches, package closure and A2 device execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from flashinfer._resources import data_root
from flashinfer.aot.build import _for_target, export_op_package
from flashinfer.jit import (
    AscendCBackend,
    AscendCToolchain,
    DiskJitBuildStore,
    JitSpecRegistry,
    declare_kernel,
    load_build_recipes,
    resolve_build_recipe,
)
from flashinfer.jit.build import _build_record
from flashinfer.operators.implementation import implementations
from flashinfer.runtime import (
    Backend,
    CapabilitySpec,
    CompileMode,
    KernelDescriptor,
    KernelLaunchABI,
    KernelRegistry,
)

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "csrc/kernels"


def test_native_declarations_close_the_selected_architecture_and_platform_tiler():
    for declaration in implementations(backend="ascendc"):
        prepared = declaration.prepare_outputs(ROOT)
        recipe = prepared["recipe"]
        architecture = {
            "Ascend910B": "arch22",
            "Ascend910C": "arch22",
            "Ascend950": "arch35",
        }[recipe.target_soc]
        assert any(architecture in Path(ref.locator).parts for ref in recipe.headers)
        if declaration.operator.name == "grouped_matmul_swiglu_quant":
            assert any("/op_host/arch22/" in ref.locator for ref in recipe.headers)
        if declaration.operator.name == "single_prefill_catlass":
            assert any("/op_host/" + architecture + "/" in ref.locator for ref in recipe.headers)
        # Resolving checks every new path/hash and the complete framework build input set.
        resolve_build_recipe(
            recipe,
            source_root=ROOT,
            toolchain_fingerprint="f" * 64,
            dependency_roots=(
                {"catlass": os.environ["FLASHINFER_CATLASS_ROOT"]}
                if recipe.dependencies
                else None
            ),
        )


@pytest.mark.parametrize(
    "entry,wrong_core,required",
    [
        (
            "gemm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp",
            310,
            "arch22",
        ),
        ("gemm/matmul/op_kernel/native.cpp", 220, "arch35"),
        ("gemm/matmul_bias_silu/op_kernel/native.cpp", 220, "arch35"),
        ("attention/flash_attention/op_kernel/catlass.cpp", 200, "arch22 or arch35"),
        ("gemm/matmul_probe/op_kernel/catlass.cpp", 200, "arch22 or arch35"),
    ],
)
def test_missing_device_implementation_fails_at_preprocessing(
    entry, wrong_core, required, tmp_path
):
    compiler = shutil.which("c++")
    assert compiler, "host C++ preprocessor is a required test dependency"
    (tmp_path / "kernel_operator.h").write_text(
        "// SDK stub: only preprocessing is exercised.\n"
    )
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/matmul_intf.h").write_text("// SDK wrapper stub for preprocessing only.\n")
    source = KERNELS / entry
    result = subprocess.run(
        [
            compiler,
            "-E",
            "-x",
            "c++",
            "-D__CCE_AICORE__=" + str(wrong_core),
            "-I" + str(tmp_path),
            "-I" + str(source.parents[1] / "include"),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "error: #error" in result.stderr and "requires " + required in result.stderr
    assert "No such file or directory" not in result.stderr


def build_probe(implementation, soc, output, cache):
    """Use the repository's public probe sources, not a generated substitute kernel."""
    from tests.multiarch_case import OPERATOR

    recipes = load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
    base = next(
        r
        for r in recipes
        if r.implementation == implementation and r.target_soc == "Ascend950"
    )
    recipe = _for_target(base, soc)
    descriptor = KernelDescriptor(
        recipe.kernel_id,
        OPERATOR.op,
        Backend.ASCENDC,
        capability=CapabilitySpec(
            OPERATOR.op,
            soc_models=(soc,),
            dtypes=("float32",),
            physical_layouts=("ND",),
        ),
        compile_mode=CompileMode.AOT,
        target_soc=soc,
        source=recipe.source,
        tiling_schema_version=2,
        launch_abi=KernelLaunchABI(
            recipe.launch_abi,
            recipe.host_entry,
            ("workspace", "workspace_bytes", "executor", "stream"),
        ),
    )
    toolchain = AscendCToolchain.discover()
    dependencies = (
        {"catlass": os.environ["FLASHINFER_CATLASS_ROOT"]}
        if recipe.dependencies
        else None
    )
    resolved = resolve_build_recipe(
        recipe,
        source_root=ROOT,
        toolchain_fingerprint=toolchain.fingerprint,
        dependency_roots=dependencies,
    )
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    spec = declare_kernel(
        recipe.kernel_id,
        backend=backend,
        registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(),
        output_path="op-package.json",
        entry_point=recipe.host_entry,
        environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=recipe.fingerprint,
    )
    store = DiskJitBuildStore(cache)
    record = _build_record(spec, backend=backend, store=store)
    path = export_op_package(store.directory(record), output)
    return descriptor, recipe, spec, path


@pytest.mark.real_compile
def test_current_csrc_native_probe_builds_and_executes_on_available_device(tmp_path):
    import torch
    import torch_npu
    from flashinfer.runtime import current_hardware
    from flashinfer.aclnn.build import build_adapter

    assert torch.npu.is_available(), "missing NPU is a failure"
    torch.npu.set_device(int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0")))
    soc = current_hardware().soc_model
    assert soc in (
        "Ascend910B",
        "Ascend910C",
        "Ascend950",
    ), "requires A2/A3/A5 hardware"
    descriptor, recipe, spec, package = build_probe(
        "native",
        soc,
        tmp_path / "release",
        os.environ.get("FLASHINFER_AOT_CACHE_DIR", tmp_path / "cache"),
    )
    # An isolated installed-metadata view keeps the production registry unchanged.
    project = tmp_path / "metadata"
    project.mkdir()
    (project / "pyproject.toml").write_text("# private installed-metadata fixture\n")
    data = data_root(project) / "kernels"
    data.mkdir(parents=True)
    (data / "registry.json").write_text(
        json.dumps(
            dict(
                schema_version=3,
                generated_at="csrc-architecture-acceptance",
                kernels=[descriptor.to_dict()],
            )
        )
    )
    (data / "ascendc_specs.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                kernels=[
                    dict(
                        kernel_id=recipe.kernel_id,
                        recipe=recipe.to_dict(),
                        spec=spec.to_dict(),
                    )
                ],
            )
        )
    )
    adapter = os.environ.get("FLASHINFER_OPERATOR_TEST_ADAPTER")
    if adapter is None:
        adapter = tmp_path / "adapter"
        build_adapter(adapter)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.multiarch_case",
            str(project),
            str(package.parent),
            str(adapter),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    (tmp_path / "native-runtime.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        '"runtime_compilations": 0' in result.stdout
        and '"numerical_cases": 2' in result.stdout
    )
    print(result.stdout)


@pytest.mark.real_compile
@pytest.mark.parametrize(
    "kernel_id",
    [
        "ascendc/gemm/matmul/native/op_package/Ascend950",
        "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950",
        "ascendc/t4/flash_attention/catlass/op_package/Ascend950",
    ],
)
def test_current_arch35_production_sources_compile_without_a5_hardware(
    kernel_id, tmp_path
):
    from flashinfer.aot.build import build_production_packages

    packages = build_production_packages(
        kernel_ids=(kernel_id,),
        output_dir=tmp_path / "release",
        cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", tmp_path / "cache"),
    )
    assert len(packages) == 1 and (packages[0] / "op-package.json").is_file()


@pytest.mark.real_compile
def test_current_arch35_catlass_probe_compiles_without_a5_hardware(tmp_path):
    _, _, _, package = build_probe(
        "catlass",
        "Ascend950",
        tmp_path / "release",
        os.environ.get("FLASHINFER_AOT_CACHE_DIR", tmp_path / "cache"),
    )
    assert (package / "op-package.json").is_file()
