"""Caller-owned FULL forward capture, with causal evidence for FIA updates.

Every case captures ONCE, with auto_dispatch_capture=False. Both the negative
control and all sequence lengths reuse that exact graph, storage and task
handle. No NPU imports or initialization occur during test collection.
"""

import pytest

pytestmark = pytest.mark.requires_soc("Ascend950")

from flashinfer.cann.attention import dense_attention
from flashinfer.npu import GraphCapability, ReplayParameterUpdates, graph_requirements
from flashinfer.runtime import SchemaError, get_global_registry


@pytest.fixture(scope="module")
def npu():
    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    if "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("requires the verified Ascend950 FIA path")
    required = ("NPUGraph", "graph", "ExternalEvent", "graph_task_group_begin",
                "graph_task_group_end", "graph_task_update_begin", "graph_task_update_end")
    if any(not hasattr(torch.npu, name) for name in required):
        pytest.skip("requires torch_npu task-group parameter updates")
    if not hasattr(torch_npu._C, "_NPUTaskGroupHandle"):
        pytest.skip("requires NPU task-group handles")
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("data", ["ramp", "random"])
def test_fia_full_capture_replay_needs_parameter_update(npu, dtype, data):
    import torch
    import torch_npu

    descriptor = get_global_registry().get("cann/attention/single_decode_fia_v1")
    assert graph_requirements(descriptor.kernel_id) == (
        GraphCapability.NEEDS_PARAM_UPDATE, ("actual_seq_lengths_kv",),
    )
    capacity, heads, kv_heads, dim = 1024, 8, 2, 64
    generator = torch.Generator().manual_seed(1101)
    if data == "ramp":
        q = torch.zeros((heads, dim), dtype=getattr(torch, dtype), device=npu)
        k = torch.zeros((capacity, kv_heads, dim), dtype=q.dtype, device=npu)
        v = torch.arange(capacity, dtype=q.dtype, device=npu)[:, None, None].expand_as(k).contiguous()
    else:
        q, k, v = (torch.randn(shape, generator=generator).to(device=npu, dtype=getattr(torch, dtype))
                   for shape in ((heads, dim), (capacity, kv_heads, dim), (capacity, kv_heads, dim)))
    # A caller allocates and owns the workspace before capture (T10 API).
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        q[None, None], k[None], v[None], num_heads=heads,
        num_key_value_heads=kv_heads, input_layout="BSND", scale=dim ** -.5,
        softmax_lse_flag=True,
    )
    out = torch.empty_like(q)
    lse = torch.empty(heads, dtype=torch.float32, device=npu)
    calls = 0

    def forward():
        nonlocal calls
        calls += 1
        result, stats = descriptor.runner(q, k, v, decode=True, return_lse=True, workspace=workspace)
        out.copy_(result)
        lse.copy_(stats)

    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    updates = ReplayParameterUpdates()
    capture_stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=False):
        with updates.recording():
            forward()  # The entire forward, including preprocessing and copies.
    captured_calls = calls
    task, = updates.tasks
    assert isinstance(task.handle, torch_npu._C._NPUTaskGroupHandle)
    assert task.layout.updates == ("actual_seq_lengths_kv",)
    assert task.descriptor is descriptor
    assert task.kernel_id == descriptor.kernel_id
    assert task.arguments["workspace"] is workspace
    assert graph.auto_dispatch_capture is False
    workspace_binding = (workspace.data_ptr(), tuple(workspace.shape))
    lengths = (63, 1, 17, 128, 513, 1024, 3, 63)
    eager = {length: tuple(value.clone() for value in dense_attention(
        q, k[:length], v[:length], decode=True, return_lse=True,
    )) for length in set(lengths) | {7, capacity}}
    torch.npu.synchronize()

    def without_update(length):
        # Disable ONLY task parameter patching. Still signal the external wait
        # node; skipping that signal would test a deadlock, not stale tiling.
        update_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(update_stream):
            task.event.record(update_stream)
        graph.replay()
        actual, actual_lse = out.cpu(), lse.cpu()
        expected, expected_lse = (value.cpu() for value in eager[length])
        error = (actual.float() - expected.float()).abs().max().item()
        assert not torch.allclose(actual, expected, atol=.01, rtol=.01)
        print(f"{dtype} {data} update=OFF seq_len={length} eager={expected[0, 0].item():.9g} "
              f"graph={actual[0, 0].item():.9g} max_abs_error={error:.9g} "
              f"lse_max_abs_error={(actual_lse - expected_lse).abs().max().item():.9g}", flush=True)

    without_update(63)
    observations = []
    # No synchronize between these updates/replays: the runtime must order
    # the previous replay before another update on the separate update stream.
    for length in lengths:
        updates.before_replay(({"kv_seq_lengths": [length]},), stream=update_stream)
        graph.replay()
        observations.append((length, out.clone(), lse.clone()))
    for length, actual, actual_lse in observations:
        expected, expected_lse = eager[length]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)
        print(f"{dtype} {data} update=ON seq_len={length} eager={expected[0, 0].item():.9g} "
              f"graph={actual[0, 0].item():.9g} "
              f"max_abs_error={(actual.float() - expected.float()).abs().max().item():.9g} "
              f"lse_max_abs_error={(actual_lse - expected_lse).abs().max().item():.9g}", flush=True)
    # After real updates, disabling them again must leave the LAST patched
    # length (63) in this same graph, rather than restoring capture length.
    without_update(7)
    torch.testing.assert_close(out, eager[63][0])
    assert calls == captured_calls
    assert (workspace.data_ptr(), tuple(workspace.shape)) == workspace_binding
    assert updates.tasks[0] is task
    print(f"{dtype} {data}: FULL captures=1 replays=10 tasks=1 "
          f"handle={type(task.handle).__name__} workspace_bytes={workspace.numel() * workspace.element_size()} "
          f"workspace_data_ptr={workspace.data_ptr()} forward_calls_during_replay={calls - captured_calls}", flush=True)


def test_updatable_fia_requires_fixed_workspace(npu):
    import torch

    q = torch.zeros((8, 64), dtype=torch.float16, device=npu)
    k = torch.zeros((128, 2, 64), dtype=q.dtype, device=npu)
    graph = torch.npu.NPUGraph()
    updates = ReplayParameterUpdates()
    # Catch inside graph() so capture is closed normally; the task recorder
    # must still invalidate its scope and clear the context on failure.
    with torch.npu.graph(graph):
        with pytest.raises(SchemaError, match="caller-owned fixed workspace"):
            with updates.recording():
                dense_attention(q, k, k, decode=True)
    assert updates.tasks == ()


@pytest.mark.parametrize("mask", ["causal", "window", "custom"])
def test_capture_time_mask_cannot_silently_follow_updated_lengths(npu, mask):
    import torch

    q = torch.zeros((8, 64), dtype=torch.float16, device=npu)
    k = torch.zeros((128, 2, 64), dtype=q.dtype, device=npu)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=npu)
    options = {"causal": True} if mask == "causal" else {"window_left": 31} if mask == "window" else {
        "custom_mask": torch.ones((1, 128), dtype=torch.bool, device=npu),
    }
    updates = ReplayParameterUpdates()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        with pytest.raises(NotImplementedError, match="unmasked attention"):
            with updates.recording():
                dense_attention(q, k, k, decode=True, workspace=workspace, **options)
    assert updates.tasks == ()
