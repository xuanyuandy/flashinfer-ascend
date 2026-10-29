"""Validate and tune CATLASS AOT versus Triton JIT in an isolated registry.

Run from the repository with an NPU Python environment and matching AOT packages.
All compilation, loading, allocation and numerical checks precede measurement.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shape_argument(value):
    try:
        s, t, h = map(int, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected S,T,H") from error
    if s not in (128, 256, 384, 512) or t not in (128, 256, 384, 512) or not 1 <= h <= 8:
        raise argparse.ArgumentTypeError("S/T must be 128/256/384/512; H must be 1..8")
    return s, t, h


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=shape_argument,
                        help="repeat S,T,H; defaults: 128,128,1 and 512,512,8")
    parser.add_argument("--dtype", nargs="+", choices=("float16", "bfloat16"), default=["float16"])
    parser.add_argument("--output-dir", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--require-both-backends", action="store_true")
    parser.add_argument("--iterations", type=int, default=200, help="invocations per timed sample")
    parser.add_argument("--warmup", type=int, default=10, help="warmup samples per candidate")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--affinity", help="optional CPU list, e.g. 16,17; applied to this process only")
    parser.add_argument("--timing-boundary", choices=("api", "runner"), default="api",
                        help="public out API or already selected backend runner; same boundary for both")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["FLASHINFER_NPU_TUNING_DB"] = str((args.output_dir / "tuning.sqlite3").resolve())
    if args.affinity:
        cpus = {int(value) for value in args.affinity.split(",")}
        if not cpus or not cpus <= os.sched_getaffinity(0):
            raise ValueError("affinity must be a nonempty subset of the allowed CPUs")
        os.sched_setaffinity(0, cpus)

    import torch
    import torch_npu  # noqa: F401
    import flashinfer
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    from flashinfer.attention.catlass_prefill import prefill_signature
    from flashinfer.runtime import current_hardware, dispatch, get_global_registry, kernel_registry_scope
    from flashinfer.triton.attention import (
        TRITON_FLASH_ATTENTION_KERNEL_ID, flash_attention_registry,
    )
    from flashinfer.tuning import (
        BenchmarkCandidate, BenchmarkConfig, collect_tuning_record, save_tuning_record,
    )

    if not torch.npu.is_available():
        raise RuntimeError("NPU is required; this verification cannot be skipped")
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.manual_seed(29)
    hardware = current_hardware()
    default = get_global_registry()
    registry = flash_attention_registry(default)
    ids = (FLASH_ATTENTION_KERNEL_ID, TRITON_FLASH_ATTENTION_KERNEL_ID)
    aot_runner = registry.get(ids[0]).runner
    aot_runner.resolve()
    if aot_runner.provenance.source != "aot":
        raise RuntimeError("comparison requires an installed CATLASS AOT package")
    print(json.dumps(dict(device_available=True, device_count=torch.npu.device_count(),
                          hardware=hardware.__dict__, default_kernels=len(default.descriptors()),
                          device=args.device,
                          isolated_kernels=len(registry.descriptors()),
                          tuning_db=os.environ["FLASHINFER_NPU_TUNING_DB"])), flush=True)
    results = []
    with kernel_registry_scope(registry):
        for dtype_name in args.dtype:
            dtype = getattr(torch, dtype_name)
            for s, t, h in args.shape or [(128, 128, 1), (512, 512, 8)]:
                cpu = [(torch.rand(n, h, 128) * 2 - 1).to(dtype) for n in (s, t, t)]
                q, k, v = [tensor.npu() for tensor in cpu]
                out = torch.empty_like(q)
                signature, _ = prefill_signature(q, k, v, out=out)
                cq, ck, cv = [tensor.double() for tensor in cpu]
                reference = torch.einsum("hst,thd->shd", torch.einsum(
                    "shd,thd->hst", cq, ck).div(128 ** .5).softmax(-1), cv)
                atol, rtol = (.001, .001) if dtype == torch.float16 else (.004, .008)
                errors = {}
                for kernel_id in ids:
                    flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=out, kernel_id=kernel_id)
                    actual = out.cpu().double()
                    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
                    errors[kernel_id] = float((actual - reference).abs().max())

                def candidate(kernel_id):
                    def prepare(iterations):
                        output = torch.empty_like(q)
                        runner = registry.get(kernel_id).runner
                        if args.timing_boundary == "runner":
                            def invoke():
                                runner(q, k, v, output)
                        else:
                            def invoke():
                                flashinfer.single_prefill_with_kv_cache_catlass(
                                    q, k, v, out=output, kernel_id=kernel_id)

                        def run():
                            for _ in range(iterations):
                                invoke()
                        return run
                    return BenchmarkCandidate(registry.get(kernel_id), prepare)

                record = collect_tuning_record(
                    signature, hardware, [candidate(kernel_id) for kernel_id in ids],
                    synchronize=torch.npu.synchronize, exact_shapes=True,
                    config=BenchmarkConfig(iterations=args.iterations, warmup=args.warmup),
                    metadata=dict(scope="isolated-registry", correctness="CPU FP64 attention",
                                  atol=atol, rtol=rtol, max_abs_error=errors,
                                  aot_spec=aot_runner.spec.fingerprint,
                                  timing_scope=args.timing_boundary + ", synchronous eager"),
                )
                save_tuning_record(record)
                label = f"{dtype_name}-{s}-{t}-{h}"
                (args.output_dir / (label + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                decision = dispatch(signature, hardware)
                triton_runner = registry.get(ids[1]).runner
                before = triton_runner.launches
                automatic = flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=out)
                torch.testing.assert_close(automatic.cpu().double(), reference, atol=atol, rtol=rtol)
                assert (triton_runner.launches == before + 1) == (decision.kernel_id == ids[1])
                # Functional and direct Torch entries must consume the same record.
                functional_signature, _ = prefill_signature(q, k, v)
                functional_decision = dispatch(functional_signature, hardware)
                assert functional_decision.kernel_id == decision.kernel_id
                functional = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)
                torch.testing.assert_close(functional.cpu().double(), reference, atol=atol, rtol=rtol)
                result = dict(shape=[s, t, h, 128], dtype=dtype_name,
                              winner=record["result"]["winner"], status=record["result"]["status"],
                              selected=decision.kernel_id, record_id=record["record_id"],
                              median_us={key: val["median_us"] for key, val in
                                         record["result"]["distributions"].items()},
                              max_abs_error=errors, result="PASS")
                results.append(result)
                print(json.dumps(result), flush=True)
    assert get_global_registry() is default
    summary = dict(results=results, default_registry_unchanged=True,
                   both_backends_selected={result["selected"] for result in results} == set(ids))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.require_both_backends and not summary["both_backends_selected"]:
        raise RuntimeError("measurements did not select both backends; evidence is retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
