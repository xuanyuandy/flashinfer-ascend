"""architectures regression coverage."""


from pathlib import Path

import pytest

from flashinfer.runtime import HardwareSignature, InputSignature, SchemaError, dispatch


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("soc", ("Ascend910B", "Ascend910C", "Ascend950"))
def test_same_public_prefill_signature_selects_the_matching_target(soc):
    request = InputSignature("ascendc.single_prefill_catlass",
        dtypes=("float16",) * 4,
        shapes=((128, 2, 128), (256, 2, 128), (256, 2, 128), (128, 2, 128)),
        physical_layouts=("ND",) * 4,
        specializations=(("tensor_contract", "supported"),))
    decision = dispatch(request, HardwareSignature(soc, 24), backend="ascendc")
    assert decision.kernel_id == "ascendc/t4/flash_attention/catlass/op_package/" + soc


def test_probe_target_mapping_updates_catlass_template_architecture():
    from flashinfer.aot.build import _for_target
    from flashinfer.jit import load_build_recipes

    recipe = next(r for r in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                  if r.implementation == "catlass" and r.target_soc == "Ascend950")
    mapped = _for_target(recipe, "Ascend910B")
    assert mapped.npu_arch == "dav-2201" and mapped.definitions == ("CATLASS_ARCH=2201",)
    assert recipe.npu_arch == "dav-3510" and recipe.definitions == ("CATLASS_ARCH=3510",)


def test_legacy_catlass_pin_cannot_be_used_by_public_v2_recipes():
    from dataclasses import replace
    from flashinfer.jit import load_build_recipes
    from flashinfer.jit.recipe import BuildDependency, _LegacyBuildDependency, _LEGACY_CATLASS_REVISION

    with pytest.raises(SchemaError, match="pinned CATLASS"):
        BuildDependency("catlass", _LEGACY_CATLASS_REVISION)
    old = _LegacyBuildDependency("catlass", _LEGACY_CATLASS_REVISION)
    recipe = next(r for r in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                  if r.implementation == "catlass")
    with pytest.raises(SchemaError, match="current pinned"):
        replace(recipe, dependencies=(old,))


import json
import os
import shutil
import subprocess
import sys


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
    from tests.support.multiarch_case import OPERATOR

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
            "tests.support.multiarch_case",
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


from flashinfer._wheel import production_kernel_ids
from flashinfer.ascendc import load_contract
from flashinfer.jit.op_package import package_target
from flashinfer.runtime import DispatchError, HardwareSignature, dispatch


@pytest.mark.parametrize("name", ["single_prefill_catlass", "grouped_matmul_swiglu_quant",
                                 "sparse_flash_attention"])
def test_arch22_variants_share_sources_and_runner_but_not_frozen_identity(name):
    variants = {d.native.target_soc: d for d in implementations(backend="ascendc")
                if d.operator.name == name}
    b, c = (variants[soc] for soc in ("Ascend910B", "Ascend910C"))
    assert b.runner_factory is c.runner_factory
    assert b.native.source_dir == c.native.source_dir
    assert b.native.build == c.native.build
    assert b.native.tensors == c.native.tensors
    db, rb, sb = load_contract(b.kernel_id)
    dc, rc, sc = load_contract(c.kernel_id)
    assert rb.source == rc.source and rb.headers == rc.headers
    assert rb.npu_arch == rc.npu_arch == "dav-2201"
    assert db.target_soc == rb.target_soc == sb.target_soc == "Ascend910B"
    assert dc.target_soc == rc.target_soc == sc.target_soc == "Ascend910C"
    assert db.fingerprint != dc.fingerprint and sb.fingerprint != sc.fingerprint
    with pytest.raises(ValueError, match="bind"):
        rb.validate_descriptor(dc)


def test_shared_wheel_inventory_is_exactly_both_physical_targets():
    expected = {d.kernel_id for d in implementations(backend="ascendc")
                if d.native.target_soc in ("Ascend910B", "Ascend910C")}
    assert set(production_kernel_ids("Ascend910B")) == expected
    assert set(production_kernel_ids("Ascend910C")) == expected
    assert len(expected) == 6


@pytest.mark.parametrize("alias", ["Ascend910C", "Ascend910_9362", "Ascend910_9391", "Ascend910_9392"])
def test_a3_compute_unit_is_not_aliased_to_a2(alias):
    assert package_target(alias) == ("ascend910_93", "dav-2201")
    assert package_target("Ascend910B3") == ("ascend910b", "dav-2201")


@pytest.mark.parametrize("soc", ["Ascend910B", "Ascend910C"])
@pytest.mark.parametrize("name", ["grouped_matmul_swiglu_quant", "sparse_flash_attention"])
def test_shared_public_semantics_route_to_only_the_physical_soc(name, soc):
    if name == "grouped_matmul_swiglu_quant":
        from tests.framework.test_grouped_matmul_swiglu_quant import inputs, request
        args, options = inputs()
        signature = request(args, options, layout="NZ")
        prefix = "ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/"
    else:
        from tests.framework.test_sparse_flash_attention import arguments, OPERATOR
        from flashinfer.operators.runtime import signature as make_signature
        signature, _, _ = make_signature(OPERATOR.op, arguments())
        prefix = "ascendc/attention/sparse_flash_attention/native/"
    hardware = HardwareSignature(soc, 24)
    assert dispatch(signature, hardware, backend="ascendc").kernel_id == prefix + soc
    other = "Ascend910C" if soc == "Ascend910B" else "Ascend910B"
    with pytest.raises(DispatchError):
        dispatch(signature, hardware, kernel_id=prefix + other)


@pytest.mark.real_compile
@pytest.mark.parametrize("name", ["grouped_matmul_swiglu_quant", "sparse_flash_attention"])
def test_a3_shared_arch22_native_package_cross_compile(name, tmp_path):
    from flashinfer.aot.build import build_production_packages
    from flashinfer.jit.cache import verify_build_directory
    from flashinfer.jit.op_package import verify_op_package

    kernel_id = next(d.kernel_id for d in implementations(backend="ascendc")
                     if d.operator.name == name and d.native.target_soc == "Ascend910C")
    release, = build_production_packages(kernel_ids=(kernel_id,), output_dir=tmp_path / "release",
        cache_dir=os.environ.get("FLASHINFER_AOT_CACHE_DIR", tmp_path / "cache"))
    _, recipe, spec = load_contract(kernel_id)
    verify_build_directory(release)
    metadata, packaged_recipe = verify_op_package(release, spec)
    assert packaged_recipe == recipe and metadata["schema_version"] == 2
    binding = json.loads((release / metadata["native_adapter"]).read_text())
    assert binding["kernel_id"] == kernel_id
    assert recipe.target_soc == "Ascend910C"
    assert any("ascend910_93" in str(p) for p in Path(release).rglob("*.json"))


def test_target_mapping_is_part_of_the_frozen_driver_identity(tmp_path, monkeypatch):
    from flashinfer.jit import ascendc

    package = tmp_path / 'flashinfer'
    driver = package / 'jit/ascendc.py'
    target_mapping = package / 'runtime/targets.py'
    driver.parent.mkdir(parents=True)
    target_mapping.parent.mkdir(parents=True)
    driver.write_text('# compiler driver\n')
    driver.with_name('op_package.py').write_text('# package target consumer\n')
    target_mapping.write_text('SOC_TARGETS = {"Ascend910C": "ascend910_93"}\n')
    monkeypatch.setattr(ascendc, '__file__', str(driver))
    before = ascendc._driver_identity()
    target_mapping.write_text('SOC_TARGETS = {"Ascend910C": "changed_compute_unit"}\n')
    assert ascendc._driver_identity() != before
