"""Reproduce T18B on A5: raw output probe, then unified Gemma RMSNorm JIT.

    python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b --probe
    python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b
    python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b --cache-only

All generated source, binaries and logs go to --cache-dir, never the repository.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(value):
    return hashlib.sha256(value).hexdigest()


def probe(root):
    directory = root / "probe"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "probe.py"
    source.write_text('''import torch, torch_npu, triton
import triton.language as tl
@triton.jit
def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < N, 0) + 1, i < N)
x = torch.arange(257, device="npu", dtype=torch.float32)
y = torch.empty_like(x)
k = add_one[(triton.cdiv(257, 128),)](x, y, N=257, BLOCK=128)
torch.npu.synchronize()
torch.testing.assert_close(y, x + 1, atol=0, rtol=0)
print("target:", k.metadata.target)
print("entry:", k.name, "runtime_resolved:", bool(k.function))
print("metadata_group:", k.metadata_group)
print("PASS add_one max_abs_error=", float((y - x - 1).abs().max()))
''', encoding="utf-8")
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(directory / "triton-cache")
    env["TRITON_ALWAYS_COMPILE"] = "1"
    log_path = directory / "probe.log"
    with log_path.open("w") as log:
        result = subprocess.run([sys.executable, str(source)], env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    print(log_path.read_text())
    result.check_returncode()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            print("FILE", path.relative_to(directory), "bytes=", len(payload), "magic=", repr(payload[:16]))
    binary = next(directory.rglob("add_one.npubin"))
    output = subprocess.check_output(["readelf", "-h", "-W", "-s", str(binary)], text=True)
    print("\n".join(line for line in output.splitlines()
                    if any(term in line for term in ("Class:", "Type:", "Machine:", "GLOBAL", "WEAK"))))


def norm(root, cache_only, rebuild_check=False):
    import torch
    import torch_npu
    import flashinfer
    from flashinfer.jit import (
        DiskJitBuildStore, JitCacheIndex, JitCompilationPolicy, JitEnvironment,
        JitSpecRegistry, TritonBackend, build_and_load, declare_kernel, describe_artifact,
    )
    from flashinfer.runtime import Backend, KernelDescriptor, KernelRegistry, SourceRef
    from flashinfer.runtime.hardware import _normalize_soc_model

    package = Path(flashinfer.__file__).parent
    locator = "triton/norm/gemma_fused_add_rmsnorm/kernels.py"
    arch = torch.npu.get_device_name(0)
    soc = _normalize_soc_model(arch)
    cann_info = Path(os.environ["ASCEND_HOME_PATH"]) / "share/info/runtime/version.info"
    environment = JitEnvironment(soc, arch, digest(cann_info.read_bytes()),
        "triton-ascend", importlib.metadata.version("triton"), torch.__version__,
        torch_npu.__version__, sys.implementation.cache_tag)
    descriptor = KernelDescriptor("triton/t18b/gemma", "norm.gemma_fused_add_rmsnorm", Backend.TRITON,
        source=SourceRef("triton", locator, digest((package / locator).read_bytes())), target_soc=soc)
    registry, specs, cache = KernelRegistry((descriptor,)), JitSpecRegistry(), JitCacheIndex()
    backend = TritonBackend()
    signature = {"x_ptr": "*fp32", "r_ptr": "*fp32", "w_ptr": "*fp32", "hidden": "i32", "eps": "fp32"}
    spec = declare_kernel(descriptor.kernel_id, backend=backend, registry=registry, spec_registry=specs,
        output_path="gemma/kernel.bin", entry_point="_gemma_fused_add_rmsnorm_kernel",
        environment_fingerprint=environment.fingerprint,
        specialization_fingerprint=digest(b"float32;BLOCK=128;hidden/eps=dynamic"),
        compile_options=("signature=" + json.dumps(signature, sort_keys=True),
                         'constexprs={"BLOCK":128}'))
    store = DiskJitBuildStore(root)
    print("disk_cache_hit:", store.lookup(spec) is not None, flush=True)
    print("declared:", spec.kernel_id, spec.fingerprint, flush=True)
    if cache_only:
        def forbidden(*args, **kwargs):
            raise AssertionError("a disk hit must not invoke any compiler subprocess")

        # A fresh process reloads the cached launcher and npubin. This catches
        # accidental compiler discovery/builds in Triton's runtime adapter too.
        subprocess.Popen = forbidden
        backend.compile = forbidden
    loaded = build_and_load(spec, backend=backend, store=store, cache=cache,
        policy=JitCompilationPolicy.CACHE_ONLY if cache_only else JitCompilationPolicy.ENABLED)
    if rebuild_check:
        before = cache.lookup(spec)
        (store.directory(before) / "unexpected-file").write_text("force a corruption miss")
        assert store.lookup(spec) is None
        loaded = build_and_load(spec, backend=backend, store=store, cache=cache)
        assert cache.lookup(spec).fingerprint == before.fingerprint
        print("PASS corruption rebuild with the same immutable JitCacheIndex", flush=True)
    torch.manual_seed(0)
    x = torch.randn((4, 128), device="npu")
    residual = torch.randn_like(x)
    weight = torch.randn((128,), device="npu")
    # CPU fp32 reference is independent of the generated NPU implementation.
    total = x.cpu() + residual.cpu()
    expected = total * torch.rsqrt(torch.mean(total * total, dim=-1, keepdim=True) + 1e-6) * (1 + weight.cpu())
    loaded.opaque_module[(4,)](x, residual, weight, 128, 1e-6)
    torch.npu.synchronize()
    torch.testing.assert_close(x.cpu(), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(residual.cpu(), total, atol=0, rtol=0)
    print("PASS gemma_fused_add_rmsnorm float32 shape=(4,128) max_abs_error=", float((x.cpu() - expected).abs().max()))
    print("PASS residual exact_equal=", torch.equal(residual.cpu(), total))
    status, = describe_artifact(descriptor.kernel_id, registry=registry, spec_registry=specs, cache=cache)
    print("status:", json.dumps({key: status.to_dict()[key] for key in (
        "is_compiled", "state", "artifact_path", "entry_point")}, sort_keys=True))
    if cache_only:
        print("PASS fresh-process cache load/launch with all subprocesses forbidden")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-check", action="store_true")
    args = parser.parse_args()
    if args.cache_only and args.rebuild_check:
        parser.error("--cache-only cannot be combined with --rebuild-check")
    root = args.cache_dir.resolve()
    if args.probe:
        probe(root)
    else:
        norm(root, args.cache_only, args.rebuild_check)


if __name__ == "__main__":
    main()
