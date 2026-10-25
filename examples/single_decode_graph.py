"""Run T3's public single-decode wrapper with a caller-owned FULL ACL graph.

From /mnt/fi/flashinfer:
  /root/miniforge3/envs/flashinfer-a5/bin/python -B \
      examples/single_decode_graph.py --dtype float16 --check-boundaries

The validated bucket is NHD, Q=(8,64), K/V=(1024,2,64), FP16/BF16,
unmasked attention, scale=0.125, and active KV lengths 1..1024.
"""

import argparse

import torch
import torch_npu  # Registers torch.npu.

from flashinfer.attention.planner import AttentionStateError
from flashinfer.decode import SingleDecodeWithKVCacheWrapper, single_decode_with_kv_cache
from flashinfer.runtime import SchemaError


def require_rejection(label, action, diagnostic):
    try:
        action()
    except (SchemaError, AttentionStateError) as error:
        if diagnostic not in str(error):
            raise AssertionError(f"{label}: missing diagnostic {diagnostic!r}: {error}") from error
        print(f"REJECT {label}: {type(error).__name__}: {error}", flush=True)
    else:
        raise AssertionError(f"{label}: operation unexpectedly accepted")


def check_boundaries(wrapper, q, k, v, update_stream):
    require_rejection("KV capacity exceeded (1025 > 1024)",
                      lambda: wrapper.before_replay(1025, stream=update_stream), "capacity")
    for name, changed in (("shape", q[:, :32]), ("batch", q.unsqueeze(0)),
                          ("dtype", q.to(torch.bfloat16 if q.dtype == torch.float16 else torch.float16))):
        require_rejection(f"frozen {name} changed", lambda: wrapper.run(changed, k, v), "new wrapper")
    require_rejection("replay without fresh parameter update", wrapper.replay, "before_replay")
    unplanned = SingleDecodeWithKVCacheWrapper()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        def capture_unplanned():
            with unplanned.recording(graph):
                unplanned.run()
        # Catch inside the graph context so the empty capture closes normally.
        require_rejection("capture without plan", capture_unplanned, "plan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check-boundaries", action="store_true")
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable: this example cannot pass by skipping")
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"Validated Ascend950 hardware required, got {name}")
    print(f"DEVICE available={torch.npu.is_available()} count={torch.npu.device_count()} "
          f"name={name} dtype={args.dtype}", flush=True)

    rng = torch.Generator().manual_seed(1101)
    q, k, v = (torch.randn(shape, generator=rng).to(device=device, dtype=getattr(torch, args.dtype))
               for shape in ((8, 64), (1024, 2, 64), (1024, 2, 64)))
    # Capture OUTSIDE: allocate public outputs and bind Q/K/V. The wrapper
    # queries and owns a fixed maximum workspace unless supplied by the caller.
    out = torch.empty_like(q)
    lse = torch.empty(8, dtype=torch.float32, device=device)
    wrapper = SingleDecodeWithKVCacheWrapper(kv_layout="NHD")
    wrapper.plan(q, k, v, kv_seq_len=1024, out=out, lse=lse)
    wrapper.warmup(iterations=3)
    print(f"PLAN kernel={wrapper.kernel_id} capacity={wrapper.kv_capacity} "
          f"workspace_bytes={wrapper.workspace_buffer.numel() * wrapper.workspace_buffer.element_size()} "
          "warmup=3", flush=True)

    # The application owns the graph, streams and capture boundary. Keep the
    # wrapper, all input/output storage, graph and streams alive together.
    graph = torch.npu.NPUGraph()
    capture_stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=False):
        with wrapper.recording(graph):
            wrapper.run()
    print("CAPTURE count=1 mode=FULL auto_dispatch_capture=False", flush=True)

    # Capture OUTSIDE: task updates and replay enqueue on the current stream.
    # replay() consumes a fresh update token, preventing stale-parameter reuse.
    lengths = (63, 1, 17, 128, 513, 1024, 3, 63)
    expected = {length: single_decode_with_kv_cache(q, k[:length], v[:length], return_lse=True)
                for length in set(lengths)}
    snapshots = []
    for length in lengths:
        wrapper.before_replay(length, stream=update_stream)
        wrapper.replay()
        snapshots.append((length, out.clone(), lse.clone()))
    torch.npu.synchronize()
    for index, (length, actual, actual_lse) in enumerate(snapshots, 1):
        eager, eager_lse = expected[length]
        torch.testing.assert_close(actual, eager, atol=0, rtol=0)
        torch.testing.assert_close(actual_lse, eager_lse, atol=0, rtol=0)
        print(f"REPLAY {index} kv_seq_len={length} "
              f"max_abs_error={(actual.float() - eager.float()).abs().max().item():.9g} "
              f"lse_max_abs_error={(actual_lse - eager_lse).abs().max().item():.9g} PASS", flush=True)
    if args.check_boundaries:
        check_boundaries(wrapper, q, k, v, update_stream)
    torch.npu.synchronize()
    print(f"PASS captures=1 replays={len(lengths)} "
          f"boundary_checks={'enabled' if args.check_boundaries else 'disabled'}", flush=True)


if __name__ == "__main__":
    main()
