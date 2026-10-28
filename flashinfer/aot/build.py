"""Release-time Ascend C op-package builds using the ordinary JitBuildStore.

python -m flashinfer.aot --config examples/ascendc_aot_native.json \
    --soc ascend950 --cache-dir /tmp/fi-cache --output-dir /tmp/fi-release
python -m flashinfer.aot --all --output-dir /tmp/fi-release
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
        raise SchemaError("this CATLASS recipe instantiates Arch::Ascend950; no recipe for " + target)
    return replace(recipe, target_soc=target, npu_arch=arch,
                   kernel_id=recipe.kernel_id.rsplit("/", 1)[0] + "/" + target)


def export_op_package(directory, output_dir):
    """Export only the public v2 operator contract into new distributions."""
    return _export_op_package(directory, output_dir, allow_legacy=False)


def _export_op_package(directory, output_dir, *, allow_legacy):
    """Copy the exact verified store generation into a read-only release layout.

    Release copies are distribution artifacts, never another build cache. Keep
    the store manifest and its full inventory; omit only its excluded build log.
    """
    directory, output_dir = Path(directory), Path(output_dir)
    record = verify_build_directory(directory)
    data = json.loads((directory / "manifest.json").read_text())
    from flashinfer.jit import JitSpec

    spec = JitSpec.from_dict(data["spec"])
    _, recipe = verify_op_package(directory, spec)
    if recipe.schema_version != 2 and not allow_legacy:
        raise SchemaError("new releases require a v2 recipe; legacy v1 artifacts are read-only compatibility")
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


def _dependencies(recipe, source_root, catlass_root=None, *, require_explicit=False):
    """Report a missing explicitly selected dependency before starting a build."""
    roots = {}
    for dependency in recipe.dependencies:
        variable = "FLASHINFER_%s_ROOT" % dependency.name.upper()
        override = catlass_root if dependency.name == "catlass" else None
        explicit = override or os.environ.get(variable)
        if require_explicit and not explicit:
            raise SchemaError(
                "%s not built due to missing dependency %s: %s is not set; "
                "set %s to the pinned checkout or pass --%s-root"
                % (recipe.kernel_id, dependency.name, variable, variable, dependency.name)
            )
        location = Path(explicit or source_root / "3rdparty" / dependency.name)
        if not (location / dependency.include_dir).is_dir():
            raise SchemaError(
                "%s not built due to missing dependency %s: headers are missing at %s; "
                "set %s to the pinned checkout (dependency headers are not bundled in the wheel)"
                % (recipe.kernel_id, dependency.name, location, variable)
            )
        roots[dependency.name] = location
    return roots


def _publish(recipe, spec, backend, store, output_dir):
    hit = store.lookup(spec) is not None
    record = _build_record(spec, backend=backend, store=store)
    output = export_op_package(store.directory(record), output_dir)
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(json.dumps(dict(kernel_id=recipe.kernel_id, recipe_id=recipe.recipe_id,
        soc=recipe.target_soc, implementation=recipe.implementation,
        output_kind=recipe.output_kind, disk_cache_hit=hit, spec=spec.fingerprint,
        package=str(output), bytes=size), sort_keys=True), flush=True)
    return output


def build_production_packages(*, output_dir, targets=(), source_root=None, cache_dir=None,
                              cann_root=None, catlass_root=None, kernel_ids=()):
    """Build every registered production AOT entry using its frozen identity.

    The installed production registry and frozen declarations are the sole
    inventory. Adding a registered contract automatically adds it to ``--all``;
    test/probe recipes are deliberately absent from the production inventory.
    Dependencies are checked for every entry before any compilation begins.
    """
    from flashinfer.ascendc import _contracts

    contracts = _contracts()
    selected = tuple(kernel_ids) if kernel_ids else tuple(contracts)
    if not selected:
        raise SchemaError("production AOT registry contains no kernels")
    unknown = set(selected) - set(contracts)
    if unknown:
        raise SchemaError("no frozen production AOT declaration for " + ", ".join(sorted(unknown)))
    root = Path(source_root) if source_root else _source_root()
    toolchain = AscendCToolchain.discover(cann_root)
    dependencies = {}
    for kernel_id in selected:
        _, recipe, _ = contracts[kernel_id]
        if targets and set(targets) != {recipe.target_soc}:
            raise SchemaError("frozen production kernel %s requires target %s; requested %s"
                              % (kernel_id, recipe.target_soc, ", ".join(targets)))
        dependencies[kernel_id] = _dependencies(recipe, root, catlass_root, require_explicit=True)
    store = DiskJitBuildStore(cache_dir)
    outputs = []
    for kernel_id in selected:
        _, recipe, spec = contracts[kernel_id]
        if toolchain.fingerprint != spec.environment_fingerprint:
            raise SchemaError(
                "installed Ascend C declaration requires its frozen build environment; "
                "current toolchain fingerprint does not match; rebuild/freeze the release "
                "before building " + kernel_id
            )
        resolved = resolve_build_recipe(recipe, source_root=root,
            toolchain_fingerprint=toolchain.fingerprint,
            dependency_roots=dependencies[kernel_id])
        if dict(spec.input_fingerprints).get("resolved_recipe") != resolved.fingerprint:
            raise SchemaError("installed Ascend C declaration does not match resolved build sources: " + kernel_id)
        backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
        outputs.append(_publish(recipe, spec, backend, store, output_dir))
    print(json.dumps(dict(summary=dict(requested=len(selected), built=len(outputs),
        failed=0, skipped=0)), sort_keys=True), flush=True)
    return tuple(outputs)


def compile_and_package(config, *, output_dir, targets=(), source_root=None, cache_dir=None,
                        cann_root=None, catlass_root=None, recipe_id=None):
    """Build a selected recipe, or all frozen production entries.

    Config ``recipe_id`` selects any recipe manifest; omitted selectors retain
    the legacy probe default. ``recipe_id=all`` uses each production entry's
    frozen implementation and exact spec, just like the ``--all`` CLI.
    """
    config = AscendCBuildConfig.from_file(config) if not isinstance(config, AscendCBuildConfig) else config
    if config.output_kind != "op_package":
        raise SchemaError("AOT requires output_kind=op_package in the BuildRecipe configuration")
    recipe_id = recipe_id or config.recipe_id
    if recipe_id == "all":
        return build_production_packages(output_dir=output_dir, targets=targets or (config.target_soc,),
            source_root=source_root, cache_dir=cache_dir, cann_root=cann_root, catlass_root=catlass_root)
    root = Path(source_root) if source_root else _source_root()
    available = tuple(recipe for path in sorted((root / "csrc/recipes").glob("*.json"))
                      for recipe in load_build_recipes(path))
    if config.target_soc.lower().startswith("ascend950"):
        available += tuple(_for_target(r, config.target_soc) for r in available
                           if r.output_kind == "op_package" and r.target_soc == "Ascend950"
                           and r.target_soc != config.target_soc)
    template = config.select(available, recipe_id)
    recipes = tuple(_for_target(template, soc) for soc in (targets or (config.target_soc,)))
    from flashinfer.ascendc import _contracts

    production = _contracts()
    if (template.kernel_id not in production
            and any(recipe.recipe_id == template.recipe_id for _, recipe, _ in production.values())):
        raise SchemaError("no frozen production AOT declaration for requested target: " + template.kernel_id)
    if template.kernel_id in production:
        if template != production[template.kernel_id][1]:
            raise SchemaError("recipe differs from frozen production declaration: " + template.kernel_id)
        return build_production_packages(output_dir=output_dir, targets=tuple(r.target_soc for r in recipes),
            source_root=root, cache_dir=cache_dir, cann_root=cann_root, catlass_root=catlass_root,
            kernel_ids=(template.kernel_id,))
    toolchain = AscendCToolchain.discover(cann_root)
    store = DiskJitBuildStore(cache_dir)
    outputs = []
    for recipe in recipes:
        resolved = resolve_build_recipe(recipe, source_root=root, toolchain_fingerprint=toolchain.fingerprint,
            dependency_roots=_dependencies(recipe, root, catlass_root))
        backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
        op = "test.t13.matmul_probe" if recipe.recipe_id == AscendCBuildConfig().recipe_id else recipe.recipe_id
        descriptor = KernelDescriptor(recipe.kernel_id, op, Backend.ASCENDC,
            compile_mode=CompileMode.AOT, target_soc=recipe.target_soc, source=recipe.source,
            tiling_schema_version=recipe.tiling_schema_version,
            launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
                                      ("workspace", "workspace_bytes", "executor", "stream")))
        spec = declare_kernel(descriptor.kernel_id, backend=backend,
            registry=KernelRegistry((descriptor,)), spec_registry=JitSpecRegistry(),
            output_path="op-package.json", entry_point=recipe.host_entry,
            environment_fingerprint=toolchain.fingerprint,
            specialization_fingerprint=hashlib.sha256(b"float32;M=16;N=16;K=16;row_major").hexdigest())
        outputs.append(_publish(recipe, spec, backend, store, output_dir))
    return tuple(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--all", action="store_true", help="build every frozen production AOT entry")
    selector.add_argument("--recipe", dest="recipe_id", help="override config recipe_id (legacy default: matmul_probe)")
    parser.add_argument("--soc", action="append", dest="targets", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--catlass-root", type=Path)
    args = vars(parser.parse_args())
    all_production = args.pop("all")
    if all_production:
        if args.pop("config") is not None:
            parser.error("select --all or --config (a config may instead use recipe_id=all)")
        args.pop("recipe_id")
        operation = build_production_packages
    else:
        if args["config"] is None:
            parser.error("supply --all or --config")
        operation = compile_and_package
    try:
        operation(**args)
    except (SchemaError, RuntimeError, OSError) as error:
        parser.exit(1, "AOT build failed: %s\n" % error)


if __name__ == "__main__":
    main()


__all__ = ["compile_and_package", "build_production_packages", "export_op_package"]
