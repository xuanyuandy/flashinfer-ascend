"""Frozen v1 probe infrastructure, excluded from public recipe entry points."""

import argparse
import hashlib
import importlib.util
from pathlib import Path

from flashinfer.jit._legacy_recipe import (
    BuildRecipe, AscendCBuildConfig, load_build_recipes, export_op_package, resolve_build_recipe,
)

ROOT = Path(__file__).resolve().parent / "fixtures/legacy_ascendc"


def prepare_jit(args):
    source = ROOT / "examples/ascendc_jit_a5.py"
    spec = importlib.util.spec_from_file_location("legacy_probe_jit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.source_root is None:
        args.source_root = ROOT
    return module.prepare(args)


def main():
    from flashinfer.aot.build import _dependencies, _for_target
    from flashinfer.jit import (AscendCBackend, AscendCToolchain, DiskJitBuildStore,
                               JitSpecRegistry, declare_kernel)
    from flashinfer.jit.build import _build_record
    from flashinfer.runtime import Backend, CompileMode, KernelDescriptor, KernelLaunchABI, KernelRegistry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--soc", default="Ascend950")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = AscendCBuildConfig.from_file(args.config)
    recipe = _for_target(config.select(load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json"),
                                       "matmul_probe"), args.soc)
    toolchain = AscendCToolchain.discover()
    resolved = resolve_build_recipe(recipe, source_root=ROOT, toolchain_fingerprint=toolchain.fingerprint,
                                    dependency_roots=_dependencies(recipe, ROOT))
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = KernelDescriptor(recipe.kernel_id, "test.t13.matmul_probe", Backend.ASCENDC,
        compile_mode=CompileMode.AOT, source=recipe.source, target_soc=recipe.target_soc,
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
                                  ("workspace", "workspace_bytes", "executor", "stream")))
    spec = declare_kernel(recipe.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="op-package.json", entry_point=recipe.host_entry,
        environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=hashlib.sha256(b"float32;M=16;N=16;K=16;row_major").hexdigest())
    store = DiskJitBuildStore(args.cache_dir)
    record = _build_record(spec, backend=backend, store=store)
    print(export_op_package(store.directory(record), args.output_dir))


if __name__ == "__main__":
    main()
