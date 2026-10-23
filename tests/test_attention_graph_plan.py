"""T9: real fixed storage and fail-closed planning, without capturing a graph."""

from dataclasses import replace

import pytest

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.attention import public_batch
from flashinfer.attention.planner import AttentionStateError
from flashinfer.runtime import Backend, SchemaError, get_global_registry
from flashinfer.runtime.registry import KernelRegistry


@pytest.fixture(scope="module")
def npu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("requires verified Ascend950")
    return torch.device("npu:0")


def index(values, device="cpu"):
    import torch

    return torch.tensor(values, dtype=torch.int32, device=device)


def make_decode(npu, *, graph=True, capacity=8):
    import torch

    buffers = (index([-7] * 3, npu), index([-7] * capacity, npu), index([-7] * 2, npu))
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(1024, dtype=torch.uint8, device=npu),
        use_cuda_graph=graph, use_tensor_cores=True,
        paged_kv_indptr_buffer=buffers[0], paged_kv_indices_buffer=buffers[1],
        paged_kv_last_page_len_buffer=buffers[2],
    )
    return wrapper, buffers


def plan_decode(wrapper, *, batch=2, q_len=1, width=None, ids=None):
    ids = list(range(batch)) if ids is None else ids
    table = None if width is None else index([[page] + [0] * (width - 1) for page in ids], wrapper._float_workspace_buffer.device)
    wrapper.plan(index(list(range(batch + 1))), index(ids), index([2] * batch),
                 4, 2, 64, 2, q_len_per_req=q_len, block_tables=table)
    return table


@pytest.mark.parametrize("mode", ["decode", "paged_prefill", "ragged_prefill"])
def test_graph_plan_copies_without_reallocating(npu, monkeypatch, mode):
    import torch

    ws = torch.empty(1024, dtype=torch.uint8, device=npu)
    if mode == "decode":
        wrapper, fixed = make_decode(npu)
        ws = wrapper._float_workspace_buffer
        first = (index([0, 1, 2]), index([0, 1]), index([1, 1]))
        second = (index([0, 2, 3]), index([2, 0, 1]), index([2, 1]))
        tail = (4, 2, 64, 2)
    elif mode == "paged_prefill":
        fixed = (index([-7] * 3, npu), index([-7] * 3, npu), index([-7] * 8, npu), index([-7] * 2, npu))
        wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, use_cuda_graph=True, qo_indptr_buf=fixed[0],
            paged_kv_indptr_buf=fixed[1], paged_kv_indices_buf=fixed[2], paged_kv_last_page_len_buf=fixed[3])
        first = (index([0, 1, 3]), index([0, 1, 2]), index([0, 1]), index([1, 1]))
        second = (index([0, 1, 3]), index([0, 2, 3]), index([2, 0, 1]), index([2, 1]))
        tail = (4, 2, 64, 2)
    else:
        fixed = (index([-7] * 3, npu), index([-7] * 3, npu))
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, use_cuda_graph=True,
                                                   qo_indptr_buf=fixed[0], kv_indptr_buf=fixed[1])
        first = (index([0, 1, 3]), index([0, 2, 5]))
        second = (index([0, 1, 3]), index([0, 3, 6]))
        tail = (4, 2, 64)
    before = tuple(value.data_ptr() for value in fixed)
    workspace_before = ws.data_ptr()

    # Even the first plan must only write existing device storage. CPU staging
    # tensors are allowed; the buffers supplied by the caller stay on NPU.
    def guard_factory(original):
        def checked(*args, **kwargs):
            device = kwargs.get("device", getattr(args[0], "device", "cpu") if args else "cpu")
            assert not str(device).startswith("npu"), "plan allocated device storage"
            return original(*args, **kwargs)
        return checked

    with monkeypatch.context() as patch:
        for name in ("tensor", "as_tensor", "empty", "zeros", "ones", "full", "arange", "empty_like"):
            patch.setattr(torch, name, guard_factory(getattr(torch, name)))
        wrapper.plan(*first, *tail)
        after_first = tuple(value.data_ptr() for value in wrapper._fixed_graph_buffers)
        wrapper.plan(*second, *tail)
        after_second = tuple(value.data_ptr() for value in wrapper._fixed_graph_buffers)
    assert before == after_first == after_second
    assert ws.data_ptr() == workspace_before
    assert all(a is b for a, b in zip(fixed, wrapper._fixed_graph_buffers))
    for buffer, source in zip(fixed, second):
        torch.testing.assert_close(buffer[:source.numel()].cpu(), source)
        assert buffer[source.numel():].tolist() == [-7] * (buffer.numel() - source.numel())
    assert wrapper.graph_resource_contract.graph_enabled
    assert wrapper.workspace_contract.graph_enabled
    assert wrapper.is_graph_enabled and wrapper.is_cuda_graph_enabled
    assert wrapper.capture_record is None
    print(f"{mode} data_ptr: before={before} plan1={after_first} plan2={after_second}; no device allocation")


@pytest.mark.parametrize("dimension", ["batch_size", "resolved_backend", "q_len_per_req", "block_table_width"])
def test_graph_plan_freezes_dimension(npu, monkeypatch, dimension):
    wrapper, fixed = make_decode(npu)
    table = plan_decode(wrapper, width=2)
    previous_plan, previous_decision = wrapper.plan_state, wrapper._native_decision
    previous_data = tuple(buffer.tolist() for buffer in fixed) + (table.tolist(),)
    options = {"width": 2}
    if dimension == "batch_size":
        options["batch"] = 1
    elif dimension == "q_len_per_req":
        options["q_len"] = 2  # Both plans satisfy use_tensor_cores=True.
    elif dimension == "block_table_width":
        options["width"] = 1  # Both tables are valid for the same paged metadata.
    else:
        resolve = public_batch.dispatch

        def changed_resolution(*args, **kwargs):
            return replace(resolve(*args, **kwargs), backend=Backend.ASCENDC)

        monkeypatch.setattr(public_batch, "dispatch", changed_resolution)
    message = "fixed batch size" if dimension == "batch_size" else "fixed " + dimension
    with pytest.raises(AttentionStateError, match=message) as error:
        plan_decode(wrapper, **options)
    assert wrapper.plan_state is previous_plan
    assert wrapper._native_decision is previous_decision
    assert tuple(buffer.tolist() for buffer in fixed) + (table.tolist(),) == previous_data
    print(f"{dimension}: {type(error.value).__name__}: {error.value}")


def test_graph_plan_rejects_another_kernel_on_the_same_backend(npu, monkeypatch):
    wrapper, _ = make_decode(npu)
    plan_decode(wrapper)
    original = get_global_registry().get(wrapper.plan_selection.kernel_id)
    alternative = replace(original, kernel_id="cann/test/another_graph_kernel", priority=100)
    registry = KernelRegistry(get_global_registry().descriptors() + (alternative,))
    dispatch = public_batch.dispatch
    monkeypatch.setattr(public_batch, "get_global_registry", lambda: registry)
    monkeypatch.setattr(public_batch, "dispatch", lambda *args, **kwargs: dispatch(*args, registry=registry, **kwargs))
    with pytest.raises(AttentionStateError, match="fixed kernel_identity") as error:
        plan_decode(wrapper)
    print(f"same backend, different kernel: {error.value}")


@pytest.mark.parametrize("first,second", [(2, 3), (3, 2), (None, 2), (2, None)])
def test_graph_plan_freezes_block_table_growth_shrink_and_presence(npu, first, second):
    wrapper, _ = make_decode(npu)
    plan_decode(wrapper, width=first)
    with pytest.raises(AttentionStateError, match="fixed block_table_width") as error:
        plan_decode(wrapper, width=second)
    print(f"block_table_width {first}->{second}: {error.value}")


def test_block_table_and_page_indices_keep_storage_and_drive_run(npu):
    import torch

    wrapper, fixed = make_decode(npu)
    table = plan_decode(wrapper, width=2)
    pointer = table.data_ptr()
    replacement = plan_decode(wrapper, width=2, ids=[1, 0])
    assert replacement is not table
    assert wrapper._fixed_block_tables is table and table.data_ptr() == pointer
    torch.testing.assert_close(table.cpu(), replacement.cpu())
    assert wrapper.graph_resource_contract.persistent_buffers[-1].name == "block_tables"
    q = torch.zeros(2, 4, 64, dtype=torch.float16, device=npu)
    keys = torch.zeros(2, 2, 2, 64, dtype=torch.float16, device=npu)
    vals = torch.stack((torch.full_like(keys[0], 3), torch.full_like(keys[0], 9)))
    torch.testing.assert_close(wrapper.run(q, (keys, vals)).cpu()[:, 0, 0], torch.tensor([9., 3.], dtype=torch.float16))
    table[:, 0].copy_(index([0, 1], npu))
    torch.testing.assert_close(wrapper.run(q, (keys, vals)).cpu()[:, 0, 0], torch.tensor([3., 9.], dtype=torch.float16))
    print(f"block_tables data_ptr: plan1={pointer} plan2={table.data_ptr()}; run reads fixed storage")

    csr, buffers = make_decode(npu)
    plan_decode(csr)
    buffers[1][:2].copy_(index([1, 0], npu))
    torch.testing.assert_close(csr.run(q, (keys, vals)).cpu()[:, 0, 0], torch.tensor([9., 3.], dtype=torch.float16))


def test_failed_first_plan_does_not_freeze_or_write(npu):
    wrapper, fixed = make_decode(npu, capacity=2)
    with pytest.raises(SchemaError, match="capacity"):
        wrapper.plan(index([0, 2, 3]), index([0, 1, 2]), index([1, 1]), 4, 2, 64, 2)
    assert not wrapper.is_planning_done
    assert wrapper._graph_plan_state is None
    assert all(buffer.eq(-7).all().item() for buffer in fixed)
    plan_decode(wrapper, q_len=2)
    with pytest.raises(AttentionStateError, match="fixed q_len_per_req"):
        plan_decode(wrapper, q_len=1)


def test_external_buffer_resize_is_rejected(npu):
    wrapper, fixed = make_decode(npu)
    plan_decode(wrapper)
    previous = wrapper.plan_state
    fixed[1].resize_(16)
    with pytest.raises(AttentionStateError, match="fixed graph buffer address/shape"):
        plan_decode(wrapper)
    assert wrapper.plan_state is previous


def test_graph_workspace_cannot_be_replaced(npu):
    import torch

    wrapper, _ = make_decode(npu)
    plan_decode(wrapper)
    with pytest.raises(AttentionStateError, match="fixed graph workspace"):
        wrapper.reset_workspace_buffer(torch.empty_like(wrapper._float_workspace_buffer),
                                       torch.empty(0, dtype=torch.uint8, device=npu))


def test_eager_can_change_batch_and_query_length(npu):
    wrapper, _ = make_decode(npu, graph=False)
    plan_decode(wrapper)
    plan_decode(wrapper, batch=1, q_len=2)
    assert wrapper.plan_state.batch_size == 1
    assert wrapper.plan_state.spec.q_len_per_req == 2
    assert wrapper._graph_plan_state is None


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_graph_prefill_custom_mask_copies_and_runs(npu, ragged, packed):
    import torch

    qo, kvptr, ids, last = (index(values, npu) for values in ([0, 0, 0], [0, 0, 0], [0, 0], [0, 0]))
    mask = torch.zeros(8, dtype=torch.uint8, device=npu)
    maskptr = index([0, 0, 0], npu)
    options = dict(use_cuda_graph=True, qo_indptr_buf=qo, custom_mask_buf=mask, mask_indptr_buf=maskptr)
    ws = torch.empty(1024, dtype=torch.uint8, device=npu)
    if ragged:
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, kv_indptr_buf=kvptr, **options)
        inputs = (index([0, 1, 2]), index([0, 2, 4]), 4, 2, 64)
    else:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, paged_kv_indptr_buf=kvptr,
            paged_kv_indices_buf=ids, paged_kv_last_page_len_buf=last, **options)
        inputs = (index([0, 1, 2]), index([0, 1, 2]), index([0, 1]), index([2, 2]), 4, 2, 64, 2)
    pointers = tuple(value.data_ptr() for value in wrapper._fixed_graph_buffers)
    q = torch.zeros(2, 4, 64, dtype=torch.float16, device=npu)
    k = torch.zeros(4, 2, 64, dtype=torch.float16, device=npu)
    v = torch.arange(4, dtype=torch.float16, device=npu)[:, None, None].expand_as(k).contiguous()
    for select in (0, 1):
        custom = (torch.tensor([1 << select] * 2, dtype=torch.uint8, device=npu) if packed else
                  torch.tensor([[select == 0, select == 1]] * 2, dtype=torch.bool, device=npu))
        wrapper.plan(*inputs, **{"packed_custom_mask" if packed else "custom_mask": custom})
        assert tuple(value.data_ptr() for value in wrapper._fixed_graph_buffers) == pointers
        assert maskptr.tolist() == ([0, 1, 2] if packed else [0, 2, 4])
        result = wrapper.run(q, k, v) if ragged else wrapper.run(q, (k.reshape(2, 2, 2, 64), v.reshape(2, 2, 2, 64)))
        torch.testing.assert_close(result.cpu()[:, 0, 0], torch.tensor([select, 2 + select], dtype=torch.float16))
