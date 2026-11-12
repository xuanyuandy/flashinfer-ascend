"""Real NPU checks for opt-in FA; invoked in fresh processes by pytest."""

import json
import os
import sys


def routing_fixture(signature, hardware, registry, winner):
    """Synthetic HOST clock tests routing ONLY; never a performance record."""
    from unittest.mock import patch
    from flashinfer.tuning import BenchmarkCandidate, BenchmarkConfig, collect_tuning_record, save_tuning_record
    from flashinfer.triton.attention import TRITON_FLASH_ATTENTION_KERNEL_ID
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID

    current, calls = [winner], [0]

    def candidate(kernel_id):
        def prepare(iterations):
            current[0] = kernel_id
            return lambda: None
        return BenchmarkCandidate(registry.get(kernel_id), prepare)

    def clock():
        calls[0] += 1
        return 0 if calls[0] % 2 else (50_000 if current[0] == winner else 100_000)

    with patch("flashinfer.tuning.benchmark.time.perf_counter_ns", clock), patch(
            "flashinfer.tuning.benchmark.time.sleep", lambda _: None):
        record = collect_tuning_record(
            signature, hardware,
            [candidate(key) for key in (FLASH_ATTENTION_KERNEL_ID, TRITON_FLASH_ATTENTION_KERNEL_ID)],
            synchronize=lambda: None, config=BenchmarkConfig(warmup=1, rounds=25, iterations=1),
            exact_shapes=True,
            metadata=dict(fixture="SYNTHETIC ROUTING UNIT TEST ONLY; NOT PERFORMANCE EVIDENCE"),
        )
    assert record["result"]["winner"] == winner
    save_tuning_record(record)


def check_automatic_routing(registry, runner):
    import torch
    import flashinfer
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    from flashinfer.attention.catlass_prefill import prefill_signature
    from flashinfer.runtime import current_hardware, dispatch
    from flashinfer.triton.attention import TRITON_FLASH_ATTENTION_KERNEL_ID

    selections = []
    for length, winner in ((128, TRITON_FLASH_ATTENTION_KERNEL_ID), (512, FLASH_ATTENTION_KERNEL_ID)):
        cpu = [(torch.rand(length, 1, 128) * 2 - 1).half() for _ in range(3)]
        q, k, v = [tensor.npu() for tensor in cpu]
        signature, _ = prefill_signature(q, k, v)
        routing_fixture(signature, current_hardware(), registry, winner)
        expected = torch.einsum("hst,thd->shd", torch.einsum(
            "shd,thd->hst", cpu[0].double(), cpu[1].double()).div(128 ** .5).softmax(-1), cpu[2].double())
        before = runner.launches
        result = flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)
        torch.testing.assert_close(result.cpu().double(), expected, atol=.001, rtol=.001)
        assert dispatch(signature, current_hardware()).kernel_id == winner
        assert (runner.launches == before + 1) == (winner == TRITON_FLASH_ATTENTION_KERNEL_ID)
        selections.append(winner)
    assert len(set(selections)) == 2
    print(json.dumps(dict(result="PASS", auto_selections=selections,
                          evidence="SYNTHETIC ROUTING FIXTURE + REAL NPU EXECUTION; NOT A PERFORMANCE RESULT")), flush=True)


def run(mode, cache_only=False):
    import torch
    import torch_npu

    if "FLASHINFER_TEST_NPU_DEVICE" in os.environ:
        torch.npu.set_device(int(os.environ["FLASHINFER_TEST_NPU_DEVICE"]))

    import flashinfer
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    from flashinfer.jit import TritonBackend
    from flashinfer.runtime import DispatchError, get_global_registry, kernel_registry_scope
    from flashinfer.triton.attention import (
        TRITON_FLASH_ATTENTION_KERNEL_ID as TRITON_ID, flash_attention_registry,
    )

    assert torch.npu.is_available() and torch.npu.device_count() == 8
    torch.set_num_threads(4)
    torch.manual_seed(29)
    if mode.startswith("queue"):
        torch_npu.npu.set_task_queue_enable(int(mode[-1]))
    default = get_global_registry()
    registry = flash_attention_registry(default)
    runner = registry.get(TRITON_ID).runner

    def forbidden(*args, **kwargs):
        raise AssertionError("a verified cache hit or warm call invoked a compiler")

    if cache_only:
        import subprocess
        subprocess.Popen = forbidden
        TritonBackend.compile = forbidden
    errors, observations = {}, []
    shapes = [(128, 128, 1), (128, 512, 8), (256, 384, 3),
              (384, 256, 2), (512, 128, 1), (512, 512, 8)] if mode == "matrix" else [(128, 256, 2)]
    stream = torch.npu.Stream()
    with kernel_registry_scope(registry), torch.npu.stream(stream):
        for dtype in (torch.float16, torch.bfloat16):
            for s, t, h in shapes:
                cpu = [(torch.rand(n + 1, h, 128) * 2 - 1).to(dtype) for n in (s, t, t)]
                q, k, v = [tensor.npu()[1:] for tensor in cpu]
                assert all(tensor.storage_offset() > 0 for tensor in (q, k, v))
                # TorchNPU producer -> Triton/ACLNN -> TorchNPU consumer.
                q = q + .125
                cq, ck, cv = ((cpu[0][1:] + .125).double(), cpu[1][1:].double(), cpu[2][1:].double())
                reference = torch.einsum("hst,thd->shd", torch.einsum(
                    "shd,thd->hst", cq, ck).div(128 ** .5).softmax(-1), cv)
                atol, rtol = (.001, .001) if dtype == torch.float16 else (.004, .008)
                for kernel_id in ((TRITON_ID,) if mode == "cold" else (FLASH_ATTENTION_KERNEL_ID, TRITON_ID)):
                    result = flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, kernel_id=kernel_id)
                    observations.append((result + .25, reference.to(dtype) + .25, atol, rtol))
                    out = torch.empty(s + 1, h, 128, dtype=dtype, device=q.device)[1:]
                    assert flashinfer.single_prefill_with_kv_cache_catlass(
                        q, k, v, out=out, kernel_id=kernel_id) is out
                    observations.append((out, reference, atol, rtol))
                with pytest_raises(DispatchError):
                    flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=q, kernel_id=TRITON_ID)
                with pytest_raises(DispatchError):
                    flashinfer.single_prefill_with_kv_cache_catlass(q, k.float(), v, kernel_id=TRITON_ID)
                with pytest_raises(DispatchError):
                    flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, causal=True, kernel_id=TRITON_ID)
        # Existing modules must remain callable without any compiler entry.
        TritonBackend.compile = forbidden
        before = runner.build_requests
        flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, kernel_id=TRITON_ID)
        assert runner.build_requests == before == 2
        if mode == "routing":
            check_automatic_routing(registry, runner)
    for i, (observed, reference, atol, rtol) in enumerate(observations):
        actual = observed.cpu().double()
        torch.testing.assert_close(actual, reference.double(), atol=atol, rtol=rtol)
        errors[str(i)] = float((actual - reference.double()).abs().max())
    assert get_global_registry() is default
    from tests.production_inventory import DEFAULT_DESCRIPTOR_COUNT
    assert len(default.descriptors()) == DEFAULT_DESCRIPTOR_COUNT
    if mode != "cold":
        assert registry.get(FLASH_ATTENTION_KERNEL_ID).runner.provenance.source == "aot"
    print(json.dumps(dict(result="PASS", mode=mode, numerical_checks=len(observations),
                          cache_only=cache_only, builds_prepared=runner.build_requests,
                          launches=runner.launches, default_registry_unchanged=True,
                          nondefault_stream=True, nonzero_storage_offset=True,
                          max_abs=max(errors.values()))), flush=True)


def pytest_raises(error):
    # This standalone script deliberately needs no pytest installation.
    from contextlib import contextmanager

    @contextmanager
    def expect():
        try:
            yield
        except error:
            return
        raise AssertionError("expected " + error.__name__)
    return expect()


if __name__ == "__main__":
    run(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "cached")
