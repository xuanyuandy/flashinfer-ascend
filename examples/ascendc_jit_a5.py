"""Real v2 A5 declaration -> compile -> ACLNN load/launch -> numerical check.

Only --config changes between native and CATLASS. This probe is registered in
an isolated registry, never added to the production builtin descriptors.
No torch, torch_npu, Triton or operator installer is needed. The common builder
produces the verified vendor tree used by the two-stage ACLNN call.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare(args):
    import flashinfer
    from flashinfer.jit import (
        AscendCBackend, AscendCBuildConfig, AscendCToolchain, JitSpec, JitSpecRegistry,
        declare_kernel, load_build_recipes, resolve_build_recipe,
    )
    from flashinfer.runtime import (
        Backend, CapabilitySpec, CompileMode, KernelDescriptor, KernelLaunchABI, KernelRegistry,
    )

    config = AscendCBuildConfig.from_file(args.config)
    variant = dict(config.overrides).get("matmul_probe", config.implementation)
    request = args.cache_dir / (variant + "-spec.json")
    if args.cache_only:
        # Real cache reuse with no compiler, source or CATLASS dependency.
        spec = JitSpec.from_dict(json.loads(request.read_text()))
        if (spec.target_soc != config.target_soc
                or spec.kernel_id != "ascendc/t13/matmul_probe/" + variant + "/op_package/" + config.target_soc):
            raise ValueError("cached request does not match the selected configuration")
        return spec, AscendCBackend()
    package = Path(flashinfer.__file__).parent
    source_root = args.source_root
    if source_root is None:
        source_root = package.parent if (package.parent / "csrc/recipes").is_dir() else package / "data/build"
    recipes = load_build_recipes(source_root / "csrc/recipes/matmul_probe.json")
    recipe = config.select(recipes, "matmul_probe")
    toolchain = AscendCToolchain.discover(args.cann_root)
    roots = {"catlass": args.catlass_root} if args.catlass_root else {}
    resolved = resolve_build_recipe(recipe, source_root=source_root,
        toolchain_fingerprint=toolchain.fingerprint, dependency_roots=roots)
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = KernelDescriptor(recipe.kernel_id, "test.t13.matmul_probe", Backend.ASCENDC,
        capability=CapabilitySpec(op="test.t13.matmul_probe", dtypes=("float32",), soc_models=("Ascend950",)),
        compile_mode=CompileMode.JIT, source=recipe.source, target_soc=recipe.target_soc,
        tiling_schema_version=recipe.tiling_schema_version,
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry,
            ("workspace", "workspace_bytes", "executor", "stream")))
    spec = declare_kernel(descriptor.kernel_id, backend=backend,
        registry=KernelRegistry((descriptor,)), spec_registry=JitSpecRegistry(),
        output_path="op-package.json", entry_point=recipe.host_entry,
        environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=hashlib.sha256(b"float32;M=16;N=16;K=16;row_major").hexdigest())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    print("recipe:", recipe.implementation, "language:", spec.source.language,
          "output:", recipe.output_kind, "dependencies:", [d.name for d in recipe.dependencies], flush=True)
    print("recipe_digest:", recipe.fingerprint, "resolved_digest:", resolved.fingerprint, flush=True)
    print("source_digests:", json.dumps({r.locator: r.digest for r in recipe.sources + recipe.headers}, sort_keys=True), flush=True)
    return spec, backend


def execute(module, entry, cann_root):
    # Both JIT and AOT consume the same verified ACLNN module and Tensor ABI.
    from examples.ascendc_aot_a5 import execute as execute_aclnn
    return execute_aclnn(module, cann_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--catlass-root", type=Path)
    parser.add_argument("--cann-root", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-check", action="store_true")
    args = parser.parse_args()
    from flashinfer.jit import DiskJitBuildStore, JitCacheIndex, JitCompilationPolicy, build_and_load

    spec, backend = prepare(args)
    store = DiskJitBuildStore(args.cache_dir)
    print("declared:", spec.kernel_id, "spec:", spec.fingerprint, flush=True)
    print("disk_cache_hit:", store.lookup(spec) is not None, flush=True)
    if args.cache_only:
        import subprocess

        def forbidden(*args, **kwargs):
            raise AssertionError("cache-only load must not probe/invoke a compiler")

        subprocess.Popen = forbidden
        backend.compile = forbidden
    cache = JitCacheIndex()
    loaded = build_and_load(spec, backend=backend, store=store, cache=cache,
        policy=JitCompilationPolicy.CACHE_ONLY if args.cache_only else JitCompilationPolicy.ENABLED)
    if args.rebuild_check:
        if args.cache_only:
            raise ValueError("--rebuild-check requires compilation enabled")
        before = store.lookup(spec)
        (store.directory(before) / "unexpected-file").write_text("corruption test\n")
        assert store.lookup(spec) is None
        loaded = build_and_load(spec, backend=backend, store=store, cache=cache)
        assert store.lookup(spec).fingerprint == before.fingerprint
        print("PASS corruption rebuild retains the same immutable cache record", flush=True)
    record = store.lookup(spec)
    print("LOAD_OK:", record.artifact.format.value, "entry:", record.artifact.entry_point, flush=True)
    print("artifact:", store.artifact_path(record), flush=True)
    result = execute(loaded.opaque_module, spec.entry_points[0], args.cann_root)
    print("PASS", json.dumps(dict(kernel=spec.kernel_id, **result), sort_keys=True), flush=True)
    assert "torch_npu" not in sys.modules and "triton" not in sys.modules


if __name__ == "__main__":
    main()
