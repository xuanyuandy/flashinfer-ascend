"""T20A: explicitly benchmark T19A BLOCK=128/256 at hidden=128 on a real NPU.

All output/database/cache paths should be outside the checkout. Example:
  ASCEND_RT_VISIBLE_DEVICES=0 python tools/t20_collect_gemma.py \
    --cache-dir /tmp/t20-jit --db /tmp/t20.sqlite3 --output /tmp/t20-quiet.json
  # Optional controlled CPU contention on the SAME affinity used for quiet:
  ... --affinity 120,121 --busy-workers 8 --output /tmp/t20-busy.json
"""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


@contextmanager
def _cpu_contention(workers):
    processes = []
    try:
        if workers:
            # Children inherit the collector's CPU affinity. A finite deadline
            # also bounds their lifetime if the collector is interrupted.
            code = ("import time\nend = time.monotonic() + 120\nx = 1\n"
                    "while time.monotonic() < end:\n x = (x * 1664525 + 1013904223) & 0xffffffff\n")
            for _ in range(workers):
                processes.append(subprocess.Popen([sys.executable, "-c", code]))
            time.sleep(2)
        yield
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=5)


def run(args):
    if args.affinity:
        os.sched_setaffinity(0, {int(cpu) for cpu in args.affinity.split(",")})
    os.environ["FLASHINFER_NPU_TUNING_DB"] = str(args.db.resolve())
    import torch
    import torch_npu
    from examples.two_level_gemma import (
        BLOCKS, ENTRY, OP, configuration_registry, device_code, digest,
    )
    from flashinfer.jit import (
        DiskJitBuildStore, JitCacheIndex, JitEnvironment, JitSpecRegistry,
        TritonBackend, build_and_load, declare_kernel,
    )
    from flashinfer.runtime import InputSignature, current_hardware, dispatch, explain_dispatch
    from flashinfer.tuning import (
        BenchmarkCandidate, BenchmarkConfig, collect_tuning_record,
        query_tuning_record, read_tuning_record, save_tuning_record,
    )

    torch.set_num_threads(1)
    torch.manual_seed(20)
    hardware = current_hardware()
    if not hardware.is_npu_available:
        raise RuntimeError("this example requires a real NPU")
    cann_info = Path(os.environ["ASCEND_HOME_PATH"]) / "share/info/runtime/version.info"
    environment = JitEnvironment(
        hardware.soc_model, torch.npu.get_device_name(0), digest(cann_info.read_bytes()),
        "triton-ascend", importlib.metadata.version("triton"), torch.__version__,
        torch_npu.__version__, sys.implementation.cache_tag,
    )
    registry = configuration_registry(hardware.soc_model)
    specs, cache, store = JitSpecRegistry(), JitCacheIndex(), DiskJitBuildStore(args.cache_dir)
    backend = TritonBackend()
    abi = {"x_ptr": "*fp32", "r_ptr": "*fp32", "w_ptr": "*fp32", "hidden": "i32", "eps": "fp32"}
    request = InputSignature(OP, dtypes=("float32",), shapes=((4, 128),))
    x0 = torch.randn((4, 128), device="npu", dtype=torch.float32)
    r0, weight = torch.randn_like(x0), torch.randn((128,), device="npu", dtype=torch.float32)
    total = x0.cpu() + r0.cpu()
    expected = total * torch.rsqrt((total * total).mean(-1, keepdim=True) + 1e-6) * (1 + weight.cpu())
    candidates, artifacts = [], {}
    for block in BLOCKS:
        kernel_id = f"triton/t19a/gemma/block{block}"
        spec = declare_kernel(
            kernel_id, backend=backend, registry=registry, spec_registry=specs,
            output_path=f"gemma/block{block}/kernel.bin", entry_point=ENTRY,
            environment_fingerprint=environment.fingerprint,
            specialization_fingerprint=digest(f"float32;BLOCK={block};hidden/eps=dynamic".encode()),
            compile_options=("signature=" + json.dumps(abi, sort_keys=True),
                             "constexprs=" + json.dumps({"BLOCK": block}, sort_keys=True)),
        )
        loaded = build_and_load(spec, backend=backend, store=store, cache=cache)
        module = loaded.opaque_module
        # Check BOTH implementations against identical pristine inputs, with the
        # in-place residual contract included. No reference timing is compared.
        x, residual = x0.clone(), r0.clone()
        module[(4,)](x, residual, weight, 128, 1e-6)
        torch.npu.synchronize()
        torch.testing.assert_close(x.cpu(), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(residual.cpu(), total, atol=0, rtol=0)
        print(f"PASS correctness {kernel_id} shape=(4,128)", flush=True)
        from flashinfer.jit import describe_artifact
        status, = describe_artifact(kernel_id, registry=registry, spec_registry=specs, cache=cache)
        payload = Path(status.artifact_path).read_bytes()
        assert digest(payload) == status.artifact.digest
        artifacts[kernel_id] = dict(spec_fingerprint=spec.fingerprint,
                                   binary_sha256=digest(payload), code_sha256=digest(device_code(payload)))

        def prepare(iterations, module=module):
            # No repeated mutation of a single input; allocation/reset excluded.
            inputs = [(x0.clone(), r0.clone()) for _ in range(iterations)]

            def execute():
                for x, residual in inputs:
                    module[(4,)](x, residual, weight, 128, 1e-6)

            return execute

        candidates.append(BenchmarkCandidate(registry.get(kernel_id), prepare))
    assert len({v["code_sha256"] for v in artifacts.values()}) == 2
    print("PASS two distinct real NPU device binaries", flush=True)
    metadata = dict(condition="controlled_cpu_contention" if args.busy_workers else "relative_idle",
                    busy_workers=args.busy_workers, python=sys.version,
                    environment=environment.to_dict() if hasattr(environment, "to_dict") else repr(environment),
                    visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), artifacts=artifacts,
                    correctness="both configurations checked against the same float32 inputs and exact residual")
    metadata["npu_smi_before"] = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=15).stdout
    with _cpu_contention(args.busy_workers):
        record = collect_tuning_record(
            request, hardware, candidates, synchronize=torch.npu.synchronize,
            config=BenchmarkConfig(warmup=args.warmup, rounds=args.rounds, iterations=args.iterations),
            metadata=metadata,
        )
    record["measurement"]["metadata"]["npu_smi_after"] = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=15).stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    save_tuning_record(record)
    assert read_tuning_record(request, hardware)["record_id"] == record["record_id"]
    assert query_tuning_record(request, hardware) == record["result"]["winner"]
    print("record:", record["record_id"], "db:", args.db, "raw:", args.output)
    print("machine:", json.dumps({
        "before_load": record["measurement"]["machine_before"]["load_average"],
        "after_load": record["measurement"]["machine_after"]["load_average"],
        "background": {k: v for k, v in record["measurement"]["machine_background"].items()
                       if k != "per_cpu_busy_percent"},
        "activity": {k: v for k, v in record["measurement"]["machine_activity"].items()
                     if k != "per_cpu_busy_percent"},
    }))
    print("samples_us:", json.dumps(record["measurement"]["samples_us"]))
    print("result:", json.dumps(record["result"]))
    print(explain_dispatch(request, hardware, registry=registry))
    print("selected:", dispatch(request, hardware, registry=registry).kernel_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "db", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--affinity", help="comma-separated Linux CPU IDs; use the same set for both conditions")
    parser.add_argument("--busy-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.busy_workers <= 16:
        parser.error("busy-workers must be between 0 and 16")
    run(args)


if __name__ == "__main__":
    main()
