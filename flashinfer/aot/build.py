"""Release-time Ascend C op-package builds using the ordinary JitBuildStore.

python -m flashinfer.aot --config examples/ascendc_aot_native.json \
    --soc ascend950 --cache-dir /tmp/fi-cache --output-dir /tmp/fi-release
No device initialization or .run installation is performed here.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from flashinfer.runtime import (
    Backend, CompileMode, KernelDescriptor, KernelLaunchABI, KernelRegistry, SchemaError,
)
from flashinfer.jit import (
    AscendCBackend, AscendCBuildConfig, AscendCToolchain, DiskJitBuildStore,
    JitSpecRegistry, declare_kernel, load_build_recipes, resolve_build_recipe,
)
from flashinfer.jit.build import _build_record
from flashinfer.jit.cache import verify_build_directory
from flashinfer.jit.op_package import package_target, verify_op_package


def _source_root():
    package = Path(__file__).resolve().parents[1]
    return package.parent if (package.parent / "csrc/recipes").is_dir() else package / "data/build"


def _for_target(recipe, target):
    compute, arch = package_target(target)
    if recipe.implementation == "catlass" and compute != "ascend950":
        raise SchemaError("this CATLASS matmul probe instantiates Arch::Ascend950; no recipe for " + target)
    return replace(recipe, target_soc=target, npu_arch=arch,
                   kernel_id=recipe.kernel_id.rsplit("/", 1)[0] + "/" + target)


def export_op_package(directory, output_dir):
    """Copy the exact verified store generation into a read-only release layout.

    Release copies are distribution artifacts, never another build cache. Keep
    the store manifest and its full inventory; omit only its excluded build log.
    """
    directory, output_dir = Path(directory), Path(output_dir)
    record = verify_build_directory(directory)
    data = json.loads((directory / "manifest.json").read_text())
    from flashinfer.jit import JitSpec

    spec = JitSpec.from_dict(data["spec"])
    verify_op_package(directory, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / spec.fingerprint
    if target.exists():
        existing = verify_build_directory(target)
        if existing != record:
            raise SchemaError("release destination has different immutable bytes: " + str(target))
        return target
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output_dir) as scratch:
        staging = Path(scratch) / spec.fingerprint
        shutil.copytree(directory, staging, ignore=shutil.ignore_patterns("build.log"))
        verify_build_directory(staging, spec=spec, record=record)
        os.rename(staging, target)
    return target


def compile_and_package(config, *, output_dir, targets=(), source_root=None, cache_dir=None,
                        cann_root=None, catlass_root=None):
    """Select BuildRecipe variants, then build/verify/publish through T18B."""
    config = AscendCBuildConfig.from_file(config) if not isinstance(config, AscendCBuildConfig) else config
    if config.output_kind != "op_package":
        raise SchemaError("AOT requires output_kind=op_package in the BuildRecipe configuration")
    root = Path(source_root) if source_root else _source_root()
    available = load_build_recipes(root / "csrc/recipes/matmul_probe.json")
    if config.target_soc.lower().startswith("ascend950"):
        available += tuple(_for_target(r, config.target_soc) for r in available
                           if r.output_kind == "op_package" and r.target_soc == "Ascend950"
                           and r.target_soc != config.target_soc)
    template = config.select(available, "matmul_probe")
    recipes = tuple(_for_target(template, soc) for soc in (targets or (config.target_soc,)))
    toolchain = AscendCToolchain.discover(cann_root)
    store = DiskJitBuildStore(cache_dir)
    outputs = []
    for recipe in recipes:
        resolved = resolve_build_recipe(recipe, source_root=root, toolchain_fingerprint=toolchain.fingerprint,
            dependency_roots={"catlass": Path(catlass_root)} if catlass_root else {})
        backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
        descriptor = KernelDescriptor(recipe.kernel_id, "test.t13.matmul_probe", Backend.ASCENDC,
            compile_mode=CompileMode.AOT, target_soc=recipe.target_soc, source=recipe.source,
            launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
                                      ("workspace", "workspace_bytes", "executor", "stream")))
        spec = declare_kernel(descriptor.kernel_id, backend=backend,
            registry=KernelRegistry((descriptor,)), spec_registry=JitSpecRegistry(),
            output_path="op-package.json", entry_point=recipe.host_entry,
            environment_fingerprint=toolchain.fingerprint,
            specialization_fingerprint=hashlib.sha256(b"float32;M=16;N=16;K=16;row_major").hexdigest())
        hit = store.lookup(spec) is not None
        record = _build_record(spec, backend=backend, store=store)
        output = export_op_package(store.directory(record), output_dir)
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        print(json.dumps(dict(soc=recipe.target_soc, implementation=recipe.implementation,
            output_kind=recipe.output_kind, disk_cache_hit=hit, spec=spec.fingerprint,
            package=str(output), bytes=size), sort_keys=True), flush=True)
        outputs.append(output)
    return tuple(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--soc", action="append", dest="targets", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--catlass-root", type=Path)
    compile_and_package(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()


__all__ = ["compile_and_package", "export_op_package"]
