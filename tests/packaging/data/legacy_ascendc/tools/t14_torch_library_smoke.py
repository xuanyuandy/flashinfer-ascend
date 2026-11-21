"""T14 evidence using a real prepared T13 artifact and PrivateUse1 execution.

JIT: --config examples/ascendc_native.json --cache-dir /tmp/t14-cache
AOT: --package /path/to/verified/release --soc ascend950
Use TORCH_LOGS=graph_breaks to also observe the deliberate positive control.
"""

import json


def prepare(args):
    from flashinfer.jit._legacy_torch_library import matmul_probe_op

    if args.package:
        from flashinfer.jit.op_package import load_op_package

        # Preserve T13-II's order: register vendor before torch_npu/ACL init.
        module = load_op_package(args.package, target_soc=args.soc)
        return matmul_probe_op(module.recipe, module), module.recipe

    from tests.support.legacy_ascendc import prepare_jit
    from flashinfer.jit import (
        DiskJitBuildStore, JitCacheIndex, JitCompilationPolicy, build_and_load,
    )
    from flashinfer.jit._legacy_recipe import BuildRecipe

    spec, backend = prepare_jit(args)
    store = DiskJitBuildStore(args.cache_dir)
    loaded = build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex(),
        policy=JitCompilationPolicy.CACHE_ONLY if args.cache_only else JitCompilationPolicy.ENABLED)
    # On cache-only loads no source tree, compiler or dependency resolution is
    # required: the build receipt already contains the exact selected recipe.
    receipt = json.loads((store.directory(store.lookup(spec)) / "ascendc-recipe.json").read_text())
    recipe = BuildRecipe.from_dict(receipt["resolved"]["recipe"])
    return matmul_probe_op(recipe, loaded, spec=spec), recipe


if __name__ == "__main__":
    from flashinfer.jit._legacy_torch_library import _MatmulProbeRunner
    from tools.check_ascendc_torch_library import check_artifact

    check_artifact(prepare, _MatmulProbeRunner)
