"""Real public-API probe for runtime AOT discovery (test fixture only).

The production registry intentionally contains no probe descriptor. This test
adapter supplies one fixed-shape declaration without putting probe knowledge in
discovery, dispatch, or the lazy runtime binder.
"""

from pathlib import Path
import hashlib
import json
import os
import sys

from flashinfer.jit import (
    AscendCBackend, AscendCToolchain, DiskJitBuildStore, JitSpec,
    JitSpecRegistry, declare_kernel, load_build_recipes, resolve_build_recipe,
)
from flashinfer.runtime import (
    Backend, CapabilitySpec, CompileMode, KernelDescriptor, KernelLaunchABI,
    KernelRegistry, ShapeRange,
)


ROOT = Path(__file__).resolve().parents[1]


def probe_descriptor(recipe):
    return KernelDescriptor(
        recipe.kernel_id, "gemm.mm_bf16", Backend.ASCENDC,
        capability=CapabilitySpec(
            op="gemm.mm_bf16", dtypes=("float32",),
            shape_ranges=tuple(ShapeRange(f"t{tensor}_axis{axis}", tensor, axis, 16, 16)
                               for tensor in range(2) for axis in range(2)),
        ),
        compile_mode=CompileMode.AOT, source=recipe.source,
        target_soc=recipe.target_soc, tiling_schema_version=recipe.tiling_schema_version,
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
                                  ("workspace", "workspace_bytes", "executor", "stream")),
    )


def build_release(root, *, target="Ascend950"):
    from flashinfer.aot.build import export_op_package
    from flashinfer.jit.build import _build_record

    recipe = next(recipe for recipe in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                  if recipe.output_kind == "op_package" and recipe.implementation == "native"
                  and recipe.target_soc == target)
    toolchain = AscendCToolchain.discover()
    resolved = resolve_build_recipe(recipe, source_root=ROOT,
                                    toolchain_fingerprint=toolchain.fingerprint)
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = probe_descriptor(recipe)
    spec = declare_kernel(
        recipe.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="op-package.json",
        entry_point=recipe.host_entry, environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=hashlib.sha256(b"p0-2;float32;16x16;row-major").hexdigest(),
    )
    store = DiskJitBuildStore(root / "build-cache")
    record = _build_record(spec, backend=backend, store=store)
    return export_op_package(store.directory(record), root / "release")


def run_scenario(scenario, declaration_release, store_root, second_release=None):
    import torch
    import torch_npu  # noqa: F401

    if "FLASHINFER_TEST_NPU_DEVICE" in os.environ:
        torch.npu.set_device(int(os.environ["FLASHINFER_TEST_NPU_DEVICE"]))

    from flashinfer import gemm
    from flashinfer.jit.discovery import AotCorruptError, AotLoadError, AotMissingError, AotPackageIndex
    from flashinfer.jit.env import JitCompilationPolicy
    from flashinfer.jit.recipe import BuildRecipe
    from flashinfer.jit.runtime import bind_aot_kernel
    from flashinfer.jit.torch_library import matmul_probe_op
    from flashinfer.runtime import registry as runtime_registry

    release = Path(declaration_release)
    manifest = json.loads((release / "manifest.json").read_text())
    spec = JitSpec.from_dict(manifest["spec"])
    recipe = BuildRecipe.from_dict(json.loads((release / "op-package.json").read_text())["recipe"])
    descriptor = probe_descriptor(recipe)
    store = DiskJitBuildStore(store_root)
    assert store.lookup(spec) is None, "scenario must start with a cold JIT store"
    counts = dict(backend_factory=0, compile=0, adapter=0)

    class CountedBackend(AscendCBackend):
        def compile(self, compilation_spec, *, build_directory):
            counts["compile"] += 1
            return super().compile(compilation_spec, build_directory=build_directory)

    def compiler():
        counts["backend_factory"] += 1
        toolchain = AscendCToolchain.discover()
        resolved = resolve_build_recipe(recipe, source_root=ROOT,
                                        toolchain_fingerprint=toolchain.fingerprint)
        assert toolchain.fingerprint == spec.environment_fingerprint
        return CountedBackend(recipes=(resolved,), toolchain=toolchain)

    def adapter(module, loaded_spec):
        counts["adapter"] += 1
        implementation = matmul_probe_op(recipe, module, spec=loaded_spec).device_impl

        def run(a, b, bias, out, out_dtype):
            assert bias is None and out_dtype == torch.float32
            if out is None:
                out = torch.empty_like(a)
            implementation(a, b, out)
            return out

        return run

    # Public callers already have real tensors: discover and activate only when
    # the selected descriptor's runner is invoked, after ACL initialization.
    torch.manual_seed(402)
    a_cpu, b_cpu = torch.randn(16, 16), torch.randn(16, 16)
    a, b = a_cpu.npu(), b_cpu.npu()
    assert torch.npu.is_initialized()
    torch.testing.assert_close((a + b).cpu(), a_cpu + b_cpu)
    policy = (JitCompilationPolicy.ENABLED if scenario in ("missing_enabled", "corrupt", "load_error")
              else JitCompilationPolicy.CACHE_ONLY if scenario == "missing_cache_only"
              else JitCompilationPolicy.DISABLED)
    bound = bind_aot_kernel(descriptor, spec=spec, runner_factory=adapter,
                            jit_backend_factory=compiler, store=store, policy=policy)
    runtime_registry._GLOBAL_REGISTRY = KernelRegistry((bound,))
    try:
        output = gemm.mm_bf16(a, b, out_dtype=torch.float32, backend="ascendc")
    except (AotCorruptError, AotLoadError, AotMissingError) as error:
        message = str(error)
        if scenario == "corrupt":
            assert isinstance(error, AotCorruptError)
            assert "digest mismatch" in message and "libcust_opapi.so" in message
        elif scenario == "load_error":
            assert isinstance(error, AotLoadError)
            assert "no ACLNN symbol" in message
            assert recipe.host_entry in message
            assert error.__cause__ is not None
        elif scenario == "incompatible":
            assert isinstance(error, AotMissingError)
            assert "all incompatible" in message and "target_soc mismatch" in message
            assert "ascend910b1" in message and "Ascend950" in message
        else:
            assert scenario in ("missing_disabled", "missing_cache_only")
            assert isinstance(error, AotMissingError)
            assert "no matching AOT op-package" in message
            assert policy.value in message
        assert counts == dict(backend_factory=0, compile=0, adapter=0), counts
        assert store.lookup(spec) is None
        print(json.dumps(dict(scenario=scenario, result="PASS", category=error.category,
                              policy=policy.value, compiler_calls=0, fallback=False,
                              diagnostic=message), sort_keys=True), flush=True)
        return

    if scenario == "identity":
        torch.testing.assert_close(output.cpu(), a_cpu @ b_cpu, atol=1e-4, rtol=1e-4)
        other = bind_aot_kernel(
            descriptor, spec=spec, runner_factory=adapter, jit_backend_factory=compiler,
            store=store, policy=JitCompilationPolicy.ENABLED,
            index=AotPackageIndex((Path(second_release),)),
        )
        runtime_registry._GLOBAL_REGISTRY = KernelRegistry((other,))
        try:
            gemm.mm_bf16(a, b, out_dtype=torch.float32, backend="ascendc")
        except AotLoadError as error:
            assert "process-scoped" in str(error) and "fresh process" in str(error)
            assert error.__cause__ is not None
            assert counts == dict(backend_factory=0, compile=0, adapter=1), counts
            assert other.runner.provenance is None
            print(json.dumps(dict(scenario=scenario, result="PASS", category=error.category,
                                  first_source=bound.runner.provenance.source, first_numerical=True,
                                  policy="enabled", compiler_calls=0, fallback=False,
                                  diagnostic=str(error)), sort_keys=True), flush=True)
            return
        raise AssertionError("second distinct package identity was not rejected")

    assert scenario in ("hit_disabled", "missing_enabled"), "expected refusal"
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), a_cpu @ b_cpu, atol=1e-4, rtol=1e-4)
    provenance = bound.runner.provenance
    expected_source = "aot" if scenario == "hit_disabled" else "jit"
    assert provenance.source == expected_source
    assert provenance.spec_fingerprint == spec.fingerprint
    assert counts["compile"] == (0 if expected_source == "aot" else 1)
    assert counts["backend_factory"] == counts["compile"]
    assert counts["adapter"] == 1
    if expected_source == "aot":
        assert Path(provenance.directory).parent == ROOT / "flashinfer/data/op_packages"
        assert store.lookup(spec) is None
    else:
        assert store.lookup(spec) is not None
        assert "no matching AOT op-package" in provenance.fallback_reason
        package_module = bound.runner._module.opaque_module
        assert str(package_module.registration_path).startswith("/proc/")
        assert package_module.registration_path.name == recipe.package_vendor
        assert package_module._registration_fd is not None
    # Subsequent public calls retain the selected module/adapter.
    again = gemm.mm_bf16(a, b, out_dtype=torch.float32, backend="ascendc")
    torch.testing.assert_close(again.cpu(), a_cpu @ b_cpu, atol=1e-4, rtol=1e-4)
    assert counts["adapter"] == 1
    print(json.dumps(dict(scenario=scenario, result="PASS", source=provenance.source,
                          policy=policy.value, compiler_calls=counts["compile"],
                          directory=str(provenance.directory),
                          late_activation=True, numerical="float32 16x16 allclose",
                          repeated_public_call=True), sort_keys=True), flush=True)


if __name__ == "__main__":
    run_scenario(*sys.argv[1:])
