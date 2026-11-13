"""T19A: two real BLOCK configurations of the T18B Triton Gemma kernel.

    ASCEND_RT_VISIBLE_DEVICES=1 python examples/two_level_gemma.py --cache-dir /tmp/t19a
    ASCEND_RT_VISIBLE_DEVICES=1 python examples/two_level_gemma.py --cache-dir /tmp/t19a --cache-only

The opt-in registry is separate from the 79 builtins. BLOCK=128 cannot cover
hidden > 128; BLOCK=256 can cover hidden <= 256. At hidden=128 both are usable,
and T7's existing kernel_id tie-break selects block128. At hidden=256 the actual
tile capacity excludes block128. No timings or performance claims are made.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from flashinfer.runtime import (
    Backend, CapabilitySpec, CompileMode, InputSignature, KernelDescriptor,
    KernelRegistry, ShapeRange, SourceRef, current_hardware, dispatch, explain_dispatch,
)


BLOCKS = (128, 256)
OP = "norm.gemma_fused_add_rmsnorm"
ENTRY = "_gemma_fused_add_rmsnorm_kernel"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def device_code(payload):
    """Return ELF executable PROGBITS sections, excluding headers/metadata."""
    from flashinfer.jit._elf import elf_symbols

    assert ENTRY in elf_symbols(payload, machine=0x1029, elf_type=2)
    header = struct.unpack_from("<HHIQQQIHHHHHH", payload, 16)
    sections = [struct.unpack_from("<IIQQQQIIQQ", payload, header[5] + i * header[10])
                for i in range(header[11])]
    code = b"".join(payload[s[4]:s[4] + s[5]] for s in sections if s[1] == 1 and s[2] & 4)
    assert code, "no executable device instructions"
    return code


def configuration_registry(soc):
    """Declare two concrete, source-backed configs; no provider imports/builds."""
    import flashinfer

    package = Path(flashinfer.__file__).parent
    locator = "triton/norm/gemma_fused_add_rmsnorm/kernels.py"
    source = SourceRef("triton", locator, digest((package / locator).read_bytes()))
    registry = KernelRegistry(
        KernelDescriptor(
            kernel_id=f"triton/t19a/gemma/block{block}", backend=Backend.TRITON,
            compile_mode=CompileMode.JIT, source=source, target_soc=soc,
            capability=CapabilitySpec(
                OP, dtypes=("float32",), soc_models=(soc,),
                shape_ranges=(ShapeRange("hidden", 0, -1, 1, block),),
            ),
        ) for block in BLOCKS
    )
    # Include a genuine alternative source so this also exercises stack
    # eligibility and opt-in explanations, without modifying the global registry.
    from flashinfer.runtime import get_global_registry

    registry.register(get_global_registry().get("reference/norm/gemma_fused_add_rmsnorm_v1"))
    registry.freeze()
    return registry


def run(root, cache_only=False):
    import torch
    import torch_npu
    from flashinfer.jit import (
        DiskJitBuildStore, JitCacheIndex, JitCompilationPolicy, JitEnvironment,
        JitSpecRegistry, TritonBackend, build_and_load, declare_kernel, describe_artifact,
    )
    arch = torch.npu.get_device_name(0)
    hardware = current_hardware()
    soc = hardware.soc_model
    cann_info = Path(os.environ["ASCEND_HOME_PATH"]) / "share/info/runtime/version.info"
    environment = JitEnvironment(
        soc, arch, digest(cann_info.read_bytes()), "triton-ascend",
        importlib.metadata.version("triton"), torch.__version__, torch_npu.__version__,
        sys.implementation.cache_tag,
    )
    registry = configuration_registry(soc)
    specs, cache, store = JitSpecRegistry(), JitCacheIndex(), DiskJitBuildStore(root)
    backend = TritonBackend()
    signature = {"x_ptr": "*fp32", "r_ptr": "*fp32", "w_ptr": "*fp32",
                 "hidden": "i32", "eps": "fp32"}
    declarations = {}
    for block in BLOCKS:
        kernel_id = f"triton/t19a/gemma/block{block}"
        declarations[kernel_id] = declare_kernel(
            kernel_id, backend=backend, registry=registry, spec_registry=specs,
            output_path=f"gemma/block{block}/kernel.bin", entry_point=ENTRY,
            environment_fingerprint=environment.fingerprint,
            specialization_fingerprint=digest(f"float32;BLOCK={block};hidden/eps=dynamic".encode()),
            compile_options=("signature=" + json.dumps(signature, sort_keys=True),
                             "constexprs=" + json.dumps({"BLOCK": block}, sort_keys=True)),
        )
    if cache_only:
        def forbidden(*args, **kwargs):
            raise AssertionError("cache-only selection/load/launch must not invoke a compiler")

        subprocess.Popen = forbidden
        backend.compile = forbidden

    torch.manual_seed(19)
    modules, artifacts = {}, {}
    selections = []

    def execute(kernel_id, hidden, label):
        # Every execution uses fresh tensors: this kernel mutates x and residual.
        x = torch.randn((4, hidden), device="npu", dtype=torch.float32)
        residual = torch.randn_like(x)
        weight = torch.randn((hidden,), device="npu", dtype=torch.float32)
        total = x.cpu() + residual.cpu()
        expected = total * torch.rsqrt((total * total).mean(-1, keepdim=True) + 1e-6) * (1 + weight.cpu())
        modules[kernel_id][(4,)](x, residual, weight, hidden, 1e-6)
        torch.npu.synchronize()
        actual, actual_residual = x.cpu(), residual.cpu()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_residual, total, atol=0, rtol=0)
        print(f"PASS {label} kernel={kernel_id} float32 shape=(4,{hidden}) "
              f"max_abs_error={float((actual - expected).abs().max())} "
              f"residual_exact={torch.equal(actual_residual, total)}", flush=True)

    for hidden in (128, 256):
        request = InputSignature(OP, dtypes=("float32",), shapes=((4, hidden),))
        decision = dispatch(request, hardware, registry=registry)
        selections.append(decision.kernel_id)
        print(explain_dispatch(request, hardware, registry=registry), flush=True)
        spec = declarations[decision.kernel_id]
        print("disk_cache_hit:", store.lookup(spec) is not None, flush=True)
        loaded = build_and_load(
            spec, backend=backend, store=store, cache=cache,
            policy=JitCompilationPolicy.CACHE_ONLY if cache_only else JitCompilationPolicy.ENABLED,
        )
        modules[decision.kernel_id] = loaded.opaque_module
        status, = describe_artifact(decision.kernel_id, registry=registry, spec_registry=specs, cache=cache)
        payload = Path(status.artifact_path).read_bytes()
        assert digest(payload) == status.artifact.digest
        artifacts[decision.kernel_id] = payload
        print("ARTIFACT", decision.kernel_id, "spec_sha256=" + spec.fingerprint,
              "npubin_sha256=" + digest(payload), "bytes=" + str(len(payload)),
              "code_sha256=" + digest(device_code(payload)),
              "code_bytes=" + str(len(device_code(payload))),
              "path=" + status.artifact_path, flush=True)
        execute(decision.kernel_id, hidden, "auto")

    assert selections == [f"triton/t19a/gemma/block{block}" for block in BLOCKS]
    assert len(set(artifacts.values())) == 2, "BLOCK options produced identical npubin bytes"
    assert len({device_code(payload) for payload in artifacts.values()}) == 2, "identical device code"
    print("PASS 2 distinct auto selections; 2 distinct real NPU ELF artifacts; device code differs", flush=True)
    # block256 is also valid where both configurations are eligible.
    execute("triton/t19a/gemma/block256", 128, "overlap")
    if cache_only:
        print("PASS fresh-process cache-only run with all subprocesses forbidden", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    run(args.cache_dir.resolve(), args.cache_only)


if __name__ == "__main__":
    main()
