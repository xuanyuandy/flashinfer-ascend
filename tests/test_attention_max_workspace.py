"""T10: byte capacity, measured eager cost and real fixed workspace use on NPU."""

import math

import pytest

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.attention.planner import AttentionStateError
from flashinfer.runtime import SchemaError
from tools.t10_max_workspace_audit import workspace_allocations


@pytest.fixture(scope="module")
def npu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("requires verified Ascend950")
    return torch.device("npu", torch.npu.current_device())


def index(values, device="cpu"):
    import torch
    return torch.tensor(values, dtype=torch.int32, device=device)


def decode(npu, *, capacity=32 * 1024 * 1024, graph=True, use_max_workspace=True, ws=None):
    import torch
    ws = torch.empty(capacity, dtype=torch.uint8, device=npu) if ws is None else ws
    return BatchDecodeWithPagedKVCacheWrapper(
        ws, use_cuda_graph=graph, use_max_workspace=use_max_workspace,
        paged_kv_indptr_buffer=index([-7, -7], npu),
        paged_kv_indices_buffer=index([-7] * 1024, npu),
        paged_kv_last_page_len_buffer=index([-7], npu),
    )


def plan(wrapper, seq_len, *, dim=64, dtype="float16"):
    pages = (seq_len + 127) // 128
    wrapper.plan(index([0, pages]), index(list(range(pages))),
                 index([0 if seq_len == 0 else (seq_len - 1) % 128 + 1]),
                 8, 2, dim, 128, q_data_type=dtype)


@pytest.mark.parametrize("dtype,dim", [("float16", 64), ("float16", 128),
                                       ("bfloat16", 64), ("bfloat16", 128)],
                         ids=["float16-64", "float16-128", "bfloat16-64", "bfloat16-128"])
def test_maximum_covers_sequence_lengths(npu, monkeypatch, dtype, dim):
    import torch
    import torch_npu

    wrapper = decode(npu)
    pointer = wrapper._float_workspace_buffer.data_ptr()
    plan(wrapper, 0, dim=dim, dtype=dtype)
    maximum = wrapper.workspace_contract.required_sizes[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("maximum workspace re-queried after sequence length changed")

    monkeypatch.setattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", forbidden)
    for seq_len in (1, 17, 128, 1024, 8192, 32768, 131072):
        plan(wrapper, seq_len, dim=dim, dtype=dtype)
        query = torch.zeros((1, 1, 8, dim), dtype=getattr(torch, dtype), device=npu)
        key = torch.zeros((1, seq_len, 2, dim), dtype=query.dtype, device=npu)
        value = torch.ones_like(key)
        eager = []
        for lse in (False, True):
            options = dict(num_heads=8, num_key_value_heads=2, input_layout="BSND",
                           scale=dim ** -.5, softmax_lse_flag=lse)
            allocations = workspace_allocations(query, key, value, **options)
            assert len(allocations) == 1, allocations
            eager.append(allocations[0])
            assert maximum >= allocations[0], (dtype, dim, seq_len, lse, maximum, allocations[0])
            assert workspace_allocations(query, key, value, workspace=wrapper._float_workspace_buffer, **options) == ()
        assert wrapper._float_workspace_buffer.data_ptr() == pointer
        assert wrapper.workspace_contract.required_sizes == (maximum, 0)
        print(f"{dtype} D={dim} seq_len={seq_len}: eager_no_lse={eager[0]} eager_lse={eager[1]} "
              f"graph={maximum} ratio={maximum / max(eager):.6f} delta={maximum - max(eager)} "
              f"data_ptr={pointer} supplied_workspace_allocations=0", flush=True)
    assert maximum == wrapper._float_workspace_buffer.numel()


@pytest.mark.parametrize("graph,enabled", [(False, False), (False, True), (True, False)],
                         ids=["eager-default", "eager-opt-in", "graph-opt-out"])
def test_eager_and_opt_out_never_query_maximum(npu, monkeypatch, graph, enabled):
    import torch
    import torch_npu
    def forbidden(*args, **kwargs):
        raise AssertionError("maximum query called without graph opt-in")
    monkeypatch.setattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", forbidden)
    wrapper = decode(npu, capacity=1024, graph=graph, use_max_workspace=enabled)
    plan(wrapper, 1)
    plan(wrapper, 17)
    assert not wrapper.workspace_contract.requirements_known
    query = torch.zeros((1, 8, 64), dtype=torch.float16, device=npu)
    key = torch.zeros((1, 128, 2, 64), dtype=query.dtype, device=npu)
    output = wrapper.run(query, (key, torch.ones_like(key)))
    torch.testing.assert_close(output, torch.ones_like(output))


def test_too_small_workspace_fails_before_freeze_or_copy(npu):
    wrapper = decode(npu, capacity=1024)
    previous = wrapper.workspace_contract
    before = [buffer.clone() for buffer in wrapper._fixed_graph_buffers]
    with pytest.raises(SchemaError, match="float workspace capacity is 1024 bytes, requires") as error:
        plan(wrapper, 1)
    assert wrapper.workspace_contract is previous
    assert not wrapper.is_planning_done and wrapper._graph_plan_state is None
    assert wrapper._native_max_workspace_key is None
    import torch
    for actual, expected in zip(wrapper._fixed_graph_buffers, before):
        torch.testing.assert_close(actual, expected)
    print("capacity:", error.value)


@pytest.mark.parametrize("mode", ["decode", "graph_decode", "paged_prefill", "ragged_prefill"])
def test_repeated_plan_and_run_use_the_bound_address(npu, monkeypatch, mode):
    import torch
    import torch_npu
    ws = torch.empty(40 * 1024 * 1024, dtype=torch.uint8, device=npu)
    if mode in ("decode", "graph_decode"):
        wrapper = (CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            ws, index([-7, -7], npu), index([-7] * 1024, npu), index([-7], npu),
            use_max_workspace=True,
        ) if mode == "graph_decode" else decode(npu, ws=ws))
        def replan(length):
            plan(wrapper, length)
    else:
        options = dict(use_cuda_graph=True, use_max_workspace=True, qo_indptr_buf=index([-7, -7], npu))
        if mode == "paged_prefill":
            wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, **options,
                paged_kv_indptr_buf=index([-7, -7], npu), paged_kv_indices_buf=index([-7], npu),
                paged_kv_last_page_len_buf=index([-7], npu))
            def replan(length):
                wrapper.plan(index([0, 1]), index([0, 1]), index([0]), index([length]), 8, 2, 64, 128)
        else:
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, **options, kv_indptr_buf=index([-7, -7], npu))
            def replan(length):
                wrapper.plan(index([0, 1]), index([0, length]), 8, 2, 64)
    original_query = torch_npu._npu_fused_infer_attention_score_get_max_workspace
    calls = []
    def query(*args, **kwargs):
        calls.append(kwargs["softmax_lse_flag"])
        return original_query(*args, **kwargs)
    monkeypatch.setattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", query)
    pointer = ws.data_ptr()
    replan(1)
    pointers = [wrapper._float_workspace_buffer.data_ptr()]
    for length in (17, 128, 3):
        replan(length)
        pointers.append(wrapper._float_workspace_buffer.data_ptr())
    assert calls == [False, True]
    assert pointers == [pointer] * 4
    assert wrapper._float_workspace_buffer is ws
    assert wrapper.workspace_contract.plan_generation == wrapper.plan_state.generation == 4
    original_out = torch_npu.npu_fused_infer_attention_score.out
    received = []
    def out(*args, **kwargs):
        received.append(kwargs["workspace"].data_ptr())
        return original_out(*args, **kwargs)
    monkeypatch.setattr(torch_npu.npu_fused_infer_attention_score, "out", out)
    q = torch.zeros((1, 8, 64), dtype=torch.float16, device=npu)
    k = torch.zeros((128, 2, 64), dtype=q.dtype, device=npu)
    v = torch.ones_like(k)
    for return_lse in (False, True):
        kv = (k[:3], v[:3]) if mode == "ragged_prefill" else (k.unsqueeze(0), v.unsqueeze(0))
        result = wrapper.run(q, *kv, return_lse=return_lse) if mode == "ragged_prefill" else wrapper.run(q, kv, return_lse=return_lse)
        output, lse = result if return_lse else (result, None)
        torch.testing.assert_close(output, torch.ones_like(output))
        if return_lse:
            torch.testing.assert_close(lse, torch.full_like(lse, math.log2(3)))
    assert received == [pointer, pointer]
    assert calls == [False, True]
    print(f"{mode} workspace data_ptr: before={pointer} plans={pointers} run.out={received}; queries={calls}")


@pytest.mark.parametrize("phase", ["plan", "run"])
def test_workspace_repoint_is_rejected(npu, phase):
    import torch
    wrapper = decode(npu)
    plan(wrapper, 1)
    previous = wrapper.plan_state
    old_pointer = wrapper._float_workspace_buffer.data_ptr()
    replacement = torch.empty_like(wrapper._float_workspace_buffer)
    wrapper._float_workspace_buffer.set_(replacement)
    assert wrapper._float_workspace_buffer.data_ptr() != old_pointer
    with pytest.raises(AttentionStateError, match="fixed graph workspace address/shape/dtype/device cannot change"):
        if phase == "plan":
            plan(wrapper, 17)
        else:
            q = torch.zeros((1, 8, 64), dtype=torch.float16, device=npu)
            k = torch.zeros((1, 128, 2, 64), dtype=q.dtype, device=npu)
            wrapper.run(q, (k, k))
    assert wrapper.plan_state is previous


@pytest.mark.parametrize("view", ["offset", "strided"])
def test_workspace_view_cannot_bypass_storage_binding(npu, view):
    import torch
    storage = torch.empty(64 * 1024 * 1024 + 1, dtype=torch.uint8, device=npu)
    ws = storage[1:] if view == "offset" else storage[:-1:2]
    wrapper = decode(npu, ws=ws)
    with pytest.raises(SchemaError, match="contiguous with storage_offset=0"):
        plan(wrapper, 1)
    assert not wrapper.is_planning_done


def test_explicit_size_query_is_read_only(npu):
    from flashinfer.npu import maximum_workspace_size
    wrapper = decode(npu, capacity=0)
    before = tuple(buffer.tolist() for buffer in wrapper._fixed_graph_buffers)
    contract = wrapper.workspace_contract
    sizes = maximum_workspace_size(wrapper, index([0, 1]), index([0]), index([1]), 8, 2, 64, 128)
    assert sizes == (32 * 1024 * 1024, 0)
    assert wrapper.workspace_contract is contract
    assert not wrapper.is_planning_done and wrapper._graph_plan_state is None
    assert wrapper._native_max_workspace_key is None
    assert tuple(buffer.tolist() for buffer in wrapper._fixed_graph_buffers) == before
    print("explicit workspace_size:", sizes, "active plan/fixed metadata unchanged")


def test_prefill_query_shape_requeries_without_new_freezes(npu, monkeypatch):
    import torch
    import torch_npu
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=npu)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, use_cuda_graph=True, use_max_workspace=True,
        qo_indptr_buf=index([-7, -7], npu), kv_indptr_buf=index([-7, -7], npu))
    original = torch_npu._npu_fused_infer_attention_score_get_max_workspace
    calls = []
    def query(*args, **kwargs):
        calls.append((args[0].shape[1], kwargs["softmax_lse_flag"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", query)
    pointer = ws.data_ptr()
    sizes = []
    for length, kv_length in ((1, 1), (3, 4096)):
        wrapper.plan(index([0, length]), index([0, kv_length]), 8, 2, 128)
        sizes.append(wrapper.workspace_contract.required_sizes[0])
        assert wrapper._float_workspace_buffer.data_ptr() == pointer
    assert calls == [(1, False), (1, True), (3, False), (3, True)]
    assert sizes[1] > sizes[0]
    print("prefill q_len 1->3:", sizes, "data_ptr:", pointer, "no additional frozen dimension")


def test_replan_larger_requirement_does_not_publish(npu):
    import torch
    ws = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=npu)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, use_cuda_graph=True, use_max_workspace=True,
        qo_indptr_buf=index([-7, -7], npu), kv_indptr_buf=index([-7, -7], npu))
    wrapper.plan(index([0, 1]), index([0, 1]), 8, 2, 128)
    previous, contract, cache_key = wrapper.plan_state, wrapper.workspace_contract, wrapper._native_max_workspace_key
    before = tuple(buffer.tolist() for buffer in wrapper._fixed_graph_buffers)
    with pytest.raises(SchemaError, match="requires 35913728"):
        wrapper.plan(index([0, 3]), index([0, 4096]), 8, 2, 128)
    assert wrapper.plan_state is previous and wrapper.workspace_contract is contract
    assert wrapper._native_max_workspace_key == cache_key
    assert tuple(buffer.tolist() for buffer in wrapper._fixed_graph_buffers) == before


@pytest.mark.parametrize("mode", ["paged", "ragged"])
@pytest.mark.parametrize("mask_kind", ["causal", "window", "custom", "packed"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_masked_prefill_uses_maximum_with_same_numerics(npu, monkeypatch, mode, mask_kind, layout):
    import torch
    import torch_npu
    torch.manual_seed(103)
    ws = torch.empty(40 * 1024 * 1024, dtype=torch.uint8, device=npu)
    options = dict(use_cuda_graph=True, use_max_workspace=True, kv_layout=layout,
                   qo_indptr_buf=index([-7, -7], npu))
    plan_options = dict(causal=mask_kind == "causal", window_left=3 if mask_kind == "window" else -1)
    if mask_kind in ("custom", "packed"):
        mask = torch.arange(51, device=npu).reshape(3, 17) % 3 != 0
        if mask_kind == "custom":
            plan_options["custom_mask"] = mask
            count = 51
        else:
            values = mask.cpu().flatten().tolist()
            packed = [sum(int(bit) << j for j, bit in enumerate(values[i:i + 8])) for i in range(0, 51, 8)]
            plan_options["packed_custom_mask"] = torch.tensor(packed, dtype=torch.uint8, device=npu)
            count = 7
        options.update(custom_mask_buf=torch.empty(count, dtype=torch.uint8, device=npu),
                       mask_indptr_buf=index([-7, -7], npu))
    if mode == "paged":
        wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, **options,
            paged_kv_indptr_buf=index([-7, -7], npu), paged_kv_indices_buf=index([-7], npu),
            paged_kv_last_page_len_buf=index([-7], npu))
        eager = BatchPrefillWithPagedKVCacheWrapper(torch.empty(0, dtype=torch.uint8, device=npu), kv_layout=layout)
        args = (index([0, 3]), index([0, 1]), index([0]), index([17]), 8, 2, 128, 32)
    else:
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws, **options, kv_indptr_buf=index([-7, -7], npu))
        eager = BatchPrefillWithRaggedKVCacheWrapper(torch.empty(0, dtype=torch.uint8, device=npu), kv_layout=layout)
        args = (index([0, 3]), index([0, 17]), 8, 2, 128)
    original_query = torch_npu._npu_fused_infer_attention_score_get_max_workspace
    masks = []
    def query(*args, **kwargs):
        masks.append((tuple(kwargs["atten_mask"].shape), kwargs["softmax_lse_flag"]))
        return original_query(*args, **kwargs)
    monkeypatch.setattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", query)
    wrapper.plan(*args, **plan_options)
    eager.plan(*args, **plan_options)
    assert masks == [((1, 1, 3, 17), False), ((1, 1, 3, 17), True)]
    q = torch.randn((3, 8, 128), dtype=torch.float16, device=npu)
    k = torch.randn((32 if mode == "paged" else 17, 2, 128), dtype=q.dtype, device=npu)
    v = torch.randn_like(k)
    if layout == "HND":
        k, v = k.transpose(0, 1), v.transpose(0, 1)
    kv = (k.unsqueeze(0), v.unsqueeze(0)) if mode == "paged" else (k, v)
    for return_lse in (False, True):
        expected = eager.run(q, kv, return_lse=return_lse) if mode == "paged" else eager.run(q, *kv, return_lse=return_lse)
        actual = wrapper.run(q, kv, return_lse=return_lse) if mode == "paged" else wrapper.run(q, *kv, return_lse=return_lse)
        torch.testing.assert_close(actual, expected)
