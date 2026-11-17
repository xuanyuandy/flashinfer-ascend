"""T14 evidence using a real prepared T13 artifact and PrivateUse1 execution.

JIT: --config examples/ascendc_native.json --cache-dir /tmp/t14-cache
AOT: --package /path/to/verified/release --soc ascend950
Use TORCH_LOGS=graph_breaks to also observe the deliberate positive control.
"""

import argparse
import json
from pathlib import Path
import sys


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


def namespace_check(torch, before):
    after = set(torch._C._dispatch_get_all_op_names())
    names = sorted(after - before)
    assert names == ["flashinfer::fused_add_rmsnorm", "flashinfer::matmul_probe", "flashinfer::rmsnorm"], names
    assert all(n.startswith("flashinfer::") and "vllm" not in n.lower()
               and "_c_ascend" not in n.lower() for n in names)
    assert not any(n.startswith("_C_ascend::") for n in after)
    for key in ("PrivateUse1", "Meta"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key("flashinfer::matmul_probe", key)
    assert not torch._C._dispatch_has_kernel_for_dispatch_key("flashinfer::matmul_probe", "CPU")
    print("NAMESPACE PASS names=" + json.dumps(names) + "; PrivateUse1=True Meta=True CPU=False", flush=True)


def memory_check(torch):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from unittest.mock import patch
    from flashinfer.jit._legacy_torch_library import _MatmulProbeRunner

    torch.npu.init()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    def counters():
        return (torch.npu.memory_allocated(), torch.npu.memory_reserved(),
                torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved())

    before = counters()
    assert before == (0, 0, 0, 0), before
    with patch.object(_MatmulProbeRunner, "__call__", side_effect=AssertionError("fake reached runtime")):
        with FakeTensorMode(allow_fallback_kernels=False):
            a, b, out = [torch.empty(16, 16, device="npu") for _ in range(3)]
            for _ in range(100):
                assert torch.ops.flashinfer.matmul_probe(a, b, out) is None
                assert isinstance(out, FakeTensor) and out.device.type == "npu"
                assert out.shape == (16, 16) and out.dtype == torch.float32
                assert counters() == (0, 0, 0, 0)
    after = counters()
    print(f"META PASS FakeTensorMode calls=100 before={before} after={after}", flush=True)


def compile_check(torch, recipe, *, positive_control=True):
    def forward(a, b, out):
        torch.ops.flashinfer.matmul_probe(a, b, out)
        return out

    a_cpu = (torch.arange(256, dtype=torch.float32) % 7 - 3).reshape(16, 16)
    b_cpu = (torch.arange(256, dtype=torch.float32) % 5 - 2).reshape(16, 16)
    expected = a_cpu @ b_cpu
    a, b = a_cpu.npu(), b_cpu.npu()
    out = torch.full_like(a, float("nan"))
    compiled = torch.compile(forward, fullgraph=True)
    for _ in range(3):
        out.fill_(float("nan"))
        assert compiled(a, b, out) is out
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    explanation = torch._dynamo.explain(forward)(a, b, out)
    targets = [str(n.target) for n in explanation.graphs[0].graph.nodes if n.op == "call_function"]
    assert explanation.graph_count == 1 and explanation.graph_break_count == 0, explanation
    assert explanation.break_reasons == [], explanation
    assert any("flashinfer.matmul_probe" in n for n in targets), targets
    print(f"COMPILE PASS output={recipe.output_kind} implementation={recipe.implementation} "
          f"backend=inductor fullgraph=True graph_count={explanation.graph_count} "
          f"graph_break_count={explanation.graph_break_count} break_reasons={explanation.break_reasons} "
          "max_abs_error=0.0 mutation_identity=True", flush=True)
    print("graph targets:", targets, flush=True)

    # Schema/fake validation also checks the actual kernel's mutation behavior.
    checks = torch.library.opcheck(torch.ops.flashinfer.matmul_probe.default, (a, b, out),
                                  test_utils=("test_schema", "test_faketensor"))
    assert set(checks.values()) == {"SUCCESS"}, checks
    print("OPCHECK PASS " + json.dumps(checks, sort_keys=True), flush=True)
    if positive_control:
        def deliberate_break(a, b, out):
            torch.ops.flashinfer.matmul_probe(a, b, out)
            torch._dynamo.graph_break()
            return out + 1

        print("GRAPH_BREAK_POSITIVE_CONTROL_BEGIN", flush=True)
        print("GRAPH_BREAK_POSITIVE_CONTROL_BEGIN", file=sys.stderr, flush=True)
        torch.compile(deliberate_break, backend="eager")(a, b, out)
        broken = torch._dynamo.explain(deliberate_break)(a, b, out)
        assert broken.graph_break_count > 0 and broken.break_reasons, broken
        try:
            torch.compile(deliberate_break, backend="eager", fullgraph=True)(a, b, out)
        except torch._dynamo.exc.Unsupported:
            pass
        else:
            raise AssertionError("fullgraph detector failed to reject deliberate break")
        print(f"POSITIVE_CONTROL PASS graph_count={broken.graph_count} "
              f"graph_break_count={broken.graph_break_count} "
              f"break_reasons={len(broken.break_reasons)} fullgraph_rejected=True", flush=True)
        print("GRAPH_BREAK_POSITIVE_CONTROL_END", flush=True)
        print("GRAPH_BREAK_POSITIVE_CONTROL_END", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--package", type=Path)
    parser.add_argument("--soc", default="ascend950")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--catlass-root", type=Path)
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--skip-positive-control", action="store_true")
    args = parser.parse_args()
    if args.config and not args.cache_dir:
        parser.error("--config requires --cache-dir")
    if args.package or args.cache_only:
        # Leave torch.compile's graph compiler available, but make any T13
        # source/dependency resolution, toolchain probing or build fatal.
        import flashinfer.jit
        from flashinfer.jit import AscendCBackend, AscendCToolchain, recipe

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared artifact attempted Ascend C resolution/build")

        AscendCToolchain.discover = forbidden
        AscendCBackend.compile = forbidden
        recipe.resolve_build_recipe = forbidden
        flashinfer.jit.resolve_build_recipe = forbidden
        print("BUILD_GUARD PASS Ascend C discovery/resolution/compilation=forbidden", flush=True)
    declaration, recipe = prepare(args)
    from flashinfer.utils import enable_torch_library
    import torch

    before = set(torch._C._dispatch_get_all_op_names())
    enable_torch_library(ascendc_ops=(declaration,))
    enable_torch_library(ascendc_ops=(declaration,))
    namespace_check(torch, before)
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("T14 evidence requires NPU")
    memory_check(torch)
    compile_check(torch, recipe, positive_control=not args.skip_positive_control)


if __name__ == "__main__":
    main()
