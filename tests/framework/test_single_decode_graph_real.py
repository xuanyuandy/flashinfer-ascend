"""T3 public single-decode graph acceptance on real Ascend950 hardware.

No low-level runner is used. Missing devices or graph APIs are failures, never
skips. All cases belong to the explicitly selected real_compile acceptance set.
"""

from contextlib import contextmanager
import os

import pytest

from flashinfer.attention.planner import AttentionStateError
from flashinfer.decode import SingleDecodeWithKVCacheWrapper, single_decode_with_kv_cache
from flashinfer.runtime import SchemaError


pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend950")]
ERRORS = (SchemaError, AttentionStateError)
LENGTHS = (63, 1, 17, 128, 513, 1024, 3, 63)


@pytest.fixture(scope="module")
def npu():
    try:
        import torch
        import torch_npu
    except ImportError as error:
        pytest.fail(f"T3 requires real Ascend950 hardware: {error}")
    assert torch.npu.is_available(), "T3 device unavailable; this is not a passing skip"
    assert torch.npu.device_count() == 8, "T3 acceptance environment requires 8 NPUs"
    index = int(os.environ.get("FLASHINFER_TEST_NPU_DEVICE", "0"))
    torch.npu.set_device(index)
    assert "950" in torch.npu.get_device_name(index), "T3 requires the validated Ascend950"
    required = ("NPUGraph", "graph", "ExternalEvent", "graph_task_group_begin",
                "graph_task_group_end", "graph_task_update_begin", "graph_task_update_end")
    assert all(hasattr(torch.npu, name) for name in required), "NPU graph APIs unavailable"
    assert hasattr(torch_npu._C, "_NPUTaskGroupHandle"), "NPU task-group handles unavailable"
    return torch.device("npu", index)


def inputs(npu, dtype="float16", data="ramp"):
    import torch

    dtype = getattr(torch, dtype)
    if data == "ramp":
        q = torch.zeros((8, 64), dtype=dtype, device=npu)
        k = torch.zeros((1024, 2, 64), dtype=dtype, device=npu)
        v = torch.arange(1024, dtype=dtype, device=npu)[:, None, None].expand_as(k).contiguous()
        return q, k, v
    rng = torch.Generator().manual_seed(1101)
    return tuple(torch.randn(shape, generator=rng).to(device=npu, dtype=dtype)
                 for shape in ((8, 64), (1024, 2, 64), (1024, 2, 64)))


def capture(wrapper):
    import torch

    graph = torch.npu.NPUGraph()
    capture_stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=False):
        with wrapper.recording(graph):
            result, lse = wrapper.run(return_lse=True)
    assert result is wrapper.output and lse is wrapper.lse
    return graph, capture_stream, update_stream


def captured(npu):
    q, k, v = inputs(npu)
    wrapper = SingleDecodeWithKVCacheWrapper()
    wrapper.plan(q, k, v)
    wrapper.warmup()
    graph, capture_stream, update_stream = capture(wrapper)
    return wrapper, q, k, v, graph, capture_stream, update_stream


@contextmanager
def rejected(label, match):
    with pytest.raises(ERRORS, match=match) as error:
        yield
    print(f"REJECT {label}: {type(error.value).__name__}: {error.value}", flush=True)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
@pytest.mark.parametrize("data", ("ramp", "random"))
def test_public_single_decode_capture_replay(npu, dtype, data):
    import torch

    q, k, v = inputs(npu, dtype, data)
    out = torch.empty_like(q)
    lse = torch.empty(8, dtype=torch.float32, device=npu)
    wrapper = SingleDecodeWithKVCacheWrapper()
    wrapper.plan(q, k, v, kv_seq_len=1024, out=out, lse=lse)
    assert wrapper.output is out and wrapper.lse is lse
    assert wrapper.kv_capacity == 1024
    assert wrapper.kernel_id == "cann/attention/single_decode_fia_v1"
    bindings = tuple(x.data_ptr() for x in (q, k, v, wrapper.workspace_buffer, out, lse))
    wrapper.warmup(iterations=3)
    graph, capture_stream, update_stream = capture(wrapper)
    assert graph.auto_dispatch_capture is False
    expected = {length: single_decode_with_kv_cache(
        q, k[:length], v[:length], return_lse=True,
    ) for length in set(LENGTHS)}
    # Copy each result on the replay stream before scheduling the next update.
    # No global synchronize between replays: event/stream ordering is exercised.
    observed = []
    for length in LENGTHS:
        wrapper.before_replay(length, stream=update_stream)
        assert wrapper.replay() is out
        observed.append((length, out.clone(), lse.clone()))
    torch.npu.synchronize()
    for length, actual, actual_lse in observed:
        eager, eager_lse = expected[length]
        torch.testing.assert_close(actual, eager, atol=0, rtol=0)
        torch.testing.assert_close(actual_lse, eager_lse, atol=0, rtol=0)
        print(f"PASS dtype={dtype} data={data} kv_seq_len={length} "
              f"max_abs_error={(actual.float() - eager.float()).abs().max().item():.9g} "
              f"lse_max_abs_error={(actual_lse - eager_lse).abs().max().item():.9g}", flush=True)
    assert tuple(x.data_ptr() for x in (q, k, v, wrapper.workspace_buffer, out, lse)) == bindings
    print(f"PASS dtype={dtype} data={data} captures=1 replays={len(LENGTHS)} "
          f"fixed_resources=True workspace_bytes="
          f"{wrapper.workspace_buffer.numel() * wrapper.workspace_buffer.element_size()}", flush=True)


def test_public_single_decode_accepts_caller_workspace(npu):
    import torch

    q, k, v = inputs(npu)
    # Obtain the provider-sized resource using the public wrapper, then supply
    # that byte capacity to a second independent wrapper before either capture.
    sized = SingleDecodeWithKVCacheWrapper()
    sized.plan(q, k, v)
    workspace_bytes = sized.workspace_buffer.numel() * sized.workspace_buffer.element_size()
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=npu)
    wrapper = SingleDecodeWithKVCacheWrapper(workspace)
    wrapper.plan(q, k, v, kv_seq_len=17)
    assert wrapper.workspace_buffer is workspace
    wrapper.warmup()
    graph, capture_stream, update_stream = capture(wrapper)
    wrapper.before_replay(513, stream=update_stream)
    out = wrapper.replay()
    eager, eager_lse = single_decode_with_kv_cache(q, k[:513], v[:513], return_lse=True)
    torch.testing.assert_close(out, eager, atol=0, rtol=0)
    torch.testing.assert_close(wrapper.lse, eager_lse, atol=0, rtol=0)
    print("PASS caller-owned workspace and initial kv_seq_len=17 -> replay=513", flush=True)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_public_single_decode_reads_updated_input_contents(npu, dtype):
    import torch

    q, k, v = inputs(npu, dtype, "random")
    wrapper = SingleDecodeWithKVCacheWrapper()
    wrapper.plan(q, k, v)
    wrapper.warmup()
    graph, capture_stream, update_stream = capture(wrapper)
    bindings = tuple(tensor.data_ptr() for tensor in (q, k, v))
    rng = torch.Generator().manual_seed(3197)
    snapshots = []
    for index, length in enumerate((1, 63, 1024, 17), 1):
        for tensor in (q, k, v):
            tensor.copy_(torch.randn(tensor.shape, generator=rng).to(device=npu, dtype=tensor.dtype))
        eager, eager_lse = single_decode_with_kv_cache(q, k[:length], v[:length], return_lse=True)
        wrapper.before_replay(length, stream=update_stream)
        wrapper.replay()
        snapshots.append((index, length, wrapper.output.clone(), wrapper.lse.clone(), eager, eager_lse))
    torch.npu.synchronize()
    for index, length, actual, actual_lse, eager, eager_lse in snapshots:
        torch.testing.assert_close(actual, eager, atol=0, rtol=0)
        torch.testing.assert_close(actual_lse, eager_lse, atol=0, rtol=0)
        print(f"PASS in-place Q/K/V dtype={dtype} round={index} kv_seq_len={length} "
              "max_abs_error=0 lse_max_abs_error=0", flush=True)
    assert tuple(tensor.data_ptr() for tensor in (q, k, v)) == bindings


def test_public_single_decode_capture_uses_only_prepared_resources(npu, monkeypatch):
    import torch
    import torch_npu
    from flashinfer.attention import single_decode_graph

    q, k, v = inputs(npu)
    wrapper = SingleDecodeWithKVCacheWrapper()
    wrapper.plan(q, k, v)
    wrapper.warmup()

    def forbidden(*args, **kwargs):
        raise AssertionError("capture/replay attempted dispatch, workspace query or output inference")

    with monkeypatch.context() as patch:
        patch.setattr(single_decode_graph, "dispatch", forbidden)
        patch.setattr(single_decode_graph, "current_hardware", forbidden)
        patch.setattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", forbidden)
        patch.setattr(torch_npu, "_npu_fused_infer_attention_score_infer_output", forbidden)
        graph, capture_stream, update_stream = capture(wrapper)
        wrapper.before_replay(63, stream=update_stream)
        wrapper.replay()
    eager, eager_lse = single_decode_with_kv_cache(q, k[:63], v[:63], return_lse=True)
    torch.testing.assert_close(wrapper.output, eager, atol=0, rtol=0)
    torch.testing.assert_close(wrapper.lse, eager_lse, atol=0, rtol=0)
    print("PASS capture/replay: no dispatch, hardware query, workspace query or output inference", flush=True)


def test_public_single_decode_rejects_insufficient_workspace(npu):
    import torch

    wrapper = SingleDecodeWithKVCacheWrapper(torch.empty(1, dtype=torch.uint8, device=npu))
    with rejected("workspace too small", "workspace capacity.*insufficient"):
        wrapper.plan(*inputs(npu))


@pytest.mark.parametrize("resource", ("q", "k", "v"))
def test_public_single_decode_rejects_output_alias(npu, resource):
    q, k, v = inputs(npu)
    aliased = {"q": q, "k": k, "v": v}[resource].view(-1)[:q.numel()].view_as(q)
    wrapper = SingleDecodeWithKVCacheWrapper()
    with rejected(f"out aliases {resource}", "out must not alias"):
        wrapper.plan(q, k, v, out=aliased)


@pytest.mark.parametrize("length", (0, 1025))
def test_public_single_decode_rejects_kv_capacity(npu, length):
    wrapper, q, k, v, graph, capture_stream, update_stream = captured(npu)
    with rejected(f"kv_seq_len={length}", "capacity"):
        wrapper.before_replay(length, stream=update_stream)
    # A failed update must not poison the graph's usable, validated range.
    wrapper.before_replay(17, stream=update_stream)
    import torch
    torch.testing.assert_close(wrapper.replay(), single_decode_with_kv_cache(q, k[:17], v[:17], return_lse=True)[0],
                               atol=0, rtol=0)


@pytest.mark.parametrize("dimension", ("shape", "batch", "dtype", "address"))
def test_public_single_decode_rejects_changed_identity(npu, dimension):
    import torch

    wrapper, q, k, v, graph, capture_stream, update_stream = captured(npu)
    changed = {
        "shape": lambda: q[:, :32],
        "batch": lambda: q.unsqueeze(0),
        "dtype": lambda: q.to(torch.bfloat16),
        "address": lambda: q.clone(),
    }[dimension]()
    with rejected(f"frozen {dimension}", "new wrapper"):
        wrapper.run(changed, k, v)
    with rejected(f"replan frozen {dimension}", "new wrapper"):
        wrapper.plan(changed, k, v)


@pytest.mark.parametrize("resource", ("q", "workspace", "out", "lse"))
def test_public_single_decode_rejects_rebound_storage(npu, resource):
    wrapper, q, k, v, graph, capture_stream, update_stream = captured(npu)
    tensor = {"q": q, "workspace": wrapper.workspace_buffer,
              "out": wrapper.output, "lse": wrapper.lse}[resource]
    old = tensor.detach()  # Keep the graph's original storage alive until exit.
    tensor.set_(tensor.clone())
    assert tensor.data_ptr() != old.data_ptr()
    with rejected(f"rebound {resource}", "new wrapper"):
        wrapper.before_replay(17, stream=update_stream)


def test_public_single_decode_rejects_missing_update(npu):
    wrapper, q, k, v, graph, capture_stream, update_stream = captured(npu)
    with rejected("first replay without update", "before_replay"):
        wrapper.replay()
    wrapper.before_replay(17, stream=update_stream)
    wrapper.replay()
    import torch
    torch.npu.synchronize()
    with rejected("second replay without fresh update", "before_replay"):
        wrapper.replay()


def test_public_single_decode_rejects_unplanned_capture(npu):
    import torch

    wrapper = SingleDecodeWithKVCacheWrapper()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        with rejected("capture without plan", "plan"):
            with wrapper.recording(graph):
                wrapper.run()


def test_public_single_decode_rejects_unwarmed_capture(npu):
    import torch

    wrapper = SingleDecodeWithKVCacheWrapper()
    wrapper.plan(*inputs(npu))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        with rejected("capture without warmup", "warmup"):
            with wrapper.recording(graph):
                wrapper.run()


def test_public_single_decode_rejects_plan_inside_capture(npu):
    import torch

    q, k, v = inputs(npu)
    wrapper = SingleDecodeWithKVCacheWrapper()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        with rejected("plan inside capture", "outside.*capture|capture.*outside"):
            wrapper.plan(q, k, v)


def test_public_single_decode_rejects_wrong_replay_stream(npu):
    import torch

    wrapper, q, k, v, graph, capture_stream, update_stream = captured(npu)
    wrapper.before_replay(17, stream=update_stream)
    with torch.npu.stream(capture_stream):
        with rejected("replay stream changed", "stream"):
            wrapper.replay()
    # Consume the pending event on its intended stream before resources die.
    wrapper.replay()
    torch.npu.synchronize()


def test_public_single_decode_rejects_two_wrappers_in_one_graph(npu):
    import torch

    q, k, v = inputs(npu)
    first, second = SingleDecodeWithKVCacheWrapper(), SingleDecodeWithKVCacheWrapper()
    for wrapper in (first, second):
        wrapper.plan(q, k, v)
        wrapper.warmup()
    graph = torch.npu.NPUGraph()
    capture_stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=False):
        with first.recording(graph):
            first.run()
        with rejected("second wrapper in same graph", "one.*wrapper.*graph"):
            with second.recording(graph):
                second.run()
    # The rejected registration must not add an unsignalled wait node or
    # invalidate the already recorded wrapper's task updates.
    first.before_replay(63, stream=update_stream)
    first.replay()
    eager, eager_lse = single_decode_with_kv_cache(q, k[:63], v[:63], return_lse=True)
    torch.testing.assert_close(first.output, eager, atol=0, rtol=0)
    torch.testing.assert_close(first.lse, eager_lse, atol=0, rtol=0)
    print("PASS first wrapper replay remains usable after rejecting second wrapper", flush=True)


def test_public_single_decode_rejects_automatic_capture_dispatch(npu):
    import torch

    wrapper = SingleDecodeWithKVCacheWrapper()
    wrapper.plan(*inputs(npu))
    wrapper.warmup()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        with rejected("automatic capture dispatch", "auto_dispatch_capture=False"):
            with wrapper.recording(graph):
                wrapper.run()


def test_public_single_decode_rejects_unconsumed_update(npu):
    import torch

    wrapper, q, k, v, graph, capture_stream, update_stream = captured(npu)
    wrapper.before_replay(17, stream=update_stream)
    # Finish only the task update and event record. No replay was enqueued,
    # so there is no external wait node to unblock before discarding the graph.
    update_stream.synchronize()
    with rejected("second update before consuming first", "new wrapper"):
        wrapper.before_replay(63, stream=update_stream)
    with rejected("replay invalidated wrapper", "new wrapper"):
        wrapper.replay()
    torch.npu.synchronize()
