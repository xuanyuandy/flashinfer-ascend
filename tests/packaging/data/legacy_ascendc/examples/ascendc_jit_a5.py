"""Real A5 declaration -> compile -> load -> ACL launch -> numerical check.

Only --config changes between native and CATLASS. This probe is registered in
an isolated registry, never added to the 79 production builtin descriptors.
No torch, torch_npu, Triton, vendor package or operator installer is needed.
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

    from flashinfer.jit._legacy_recipe import AscendCBuildConfig, load_build_recipes, resolve_build_recipe

    config = AscendCBuildConfig.from_file(args.config)
    variant = dict(config.overrides).get("matmul_probe", config.implementation)
    request = args.cache_dir / (variant + "-spec.json")
    if args.cache_only:
        # Real cache reuse with no compiler, source or CATLASS dependency.
        spec = JitSpec.from_dict(json.loads(request.read_text()))
        if (spec.target_soc != config.target_soc
                or spec.kernel_id != "ascendc/t13/matmul_probe/" + variant + "/" + config.target_soc):
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
        launch_abi=KernelLaunchABI(recipe.launch_abi, recipe.host_entry, ("stream", "a", "b", "c")))
    spec = declare_kernel(descriptor.kernel_id, backend=backend,
        registry=KernelRegistry((descriptor,)), spec_registry=JitSpecRegistry(),
        output_path="matmul_probe/kernel.so", entry_point=recipe.host_entry,
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
    # Loading the runtime is explicit and happens after the build/load receipt.
    acl = C.CDLL(str(cann_root / "lib64/libascendcl.so"), mode=C.RTLD_GLOBAL)
    prototypes = {
        "aclInit": [C.c_char_p], "aclFinalize": [],
        "aclrtSetDevice": [C.c_int], "aclrtResetDevice": [C.c_int],
        "aclrtCreateStream": [C.POINTER(C.c_void_p)], "aclrtDestroyStream": [C.c_void_p],
        "aclrtSynchronizeStream": [C.c_void_p],
        "aclrtMalloc": [C.POINTER(C.c_void_p), C.c_size_t, C.c_int], "aclrtFree": [C.c_void_p],
        "aclrtMemcpy": [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int],
    }
    for name, arguments in prototypes.items():
        function = getattr(acl, name)
        function.argtypes, function.restype = arguments, C.c_int

    def call(name, *arguments):
        result = getattr(acl, name)(*arguments)
        if result:
            raise RuntimeError("%s returned %s" % (name, result))

    launcher = getattr(module, entry)
    launcher.argtypes, launcher.restype = [C.c_void_p] * 4, None
    call("aclInit", None)
    stream = C.c_void_p()
    allocations = []
    try:
        call("aclrtSetDevice", 0)
        call("aclrtCreateStream", C.byref(stream))
        array = C.c_float * 256
        ha = array(*(float(i % 7 - 3) for i in range(256)))
        hb = array(*(float(i % 5 - 2) for i in range(256)))
        hc = array(*([float("nan")] * 256))
        for host in (ha, hb, hc):
            pointer = C.c_void_p()
            call("aclrtMalloc", C.byref(pointer), C.sizeof(host), 0)
            allocations.append(pointer)
            call("aclrtMemcpy", pointer, C.sizeof(host), C.cast(host, C.c_void_p), C.sizeof(host), 1)
        launcher(stream, *allocations)
        call("aclrtSynchronizeStream", stream)
        call("aclrtMemcpy", C.cast(hc, C.c_void_p), C.sizeof(hc), allocations[2], C.sizeof(hc), 2)
        expected = [sum(ha[m * 16 + k] * hb[k * 16 + n] for k in range(16))
                    for m in range(16) for n in range(16)]
        # NaN-filled output catches a no-op launcher as well as wrong results.
        if not all(hc[i] == expected[i] for i in range(256)):
            raise AssertionError("NPU output differs from independent CPU GEMM")
        return dict(shape=[16, 16, 16], device=0, stream_sync=0,
                    max_abs_error=max(abs(hc[i] - expected[i]) for i in range(256)), first8=list(hc)[:8])
    finally:
        for pointer in reversed(allocations):
            call("aclrtFree", pointer)
        if stream.value:
            call("aclrtDestroyStream", stream)
        call("aclrtResetDevice", 0)
        call("aclFinalize")


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
