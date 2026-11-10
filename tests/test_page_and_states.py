import math

import pytest
import torch

import flashinfer


@pytest.fixture
def npu(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("requires verified Ascend950")
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "cann")
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("states", [1, 3, 17, 100])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_merge_states_native_tensor_list_log2_and_empty_support(npu, states, dtype):
    torch.manual_seed(21)
    v = torch.randn(3, states, 2, 64, dtype=dtype)
    s = torch.randn(3, states, 2)
    s[0, :, 0] = -math.inf
    if states > 15:
        s[1, :15, 1] = -math.inf
    mask = s.isfinite().any(1)
    safe = torch.where(mask[:, None], s, 0)
    weights = torch.softmax(safe * math.log(2), 1)
    expected = (v.float() * weights[..., None]).sum(1).to(dtype)
    expected = torch.where(mask[..., None], expected, 0)
    expected_lse = (safe * math.log(2)).logsumexp(1) / math.log(2)
    expected_lse.masked_fill_(~mask, -math.inf)
    output, lse = flashinfer.merge_states(v.to(npu), s.to(npu))
    torch.testing.assert_close(output.cpu(), expected, rtol=0.015, atol=0.015)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_page_append_slots_aliases_and_untouched_values(npu, layout, packed, dtype):
    keys = torch.full((4, 4, 2, 64), -99, dtype=dtype)
    values = torch.full_like(keys, -88)
    append_key = torch.arange(3 * 2 * 64).reshape(3, 2, 64).to(dtype)
    append_value = -append_key
    expected_k, expected_v = keys.clone(), values.clone()
    for token, (page, offset) in enumerate(((2, 1), (1, 2), (0, 1))):
        expected_k[page, offset] = append_key[token]
        expected_v[page, offset] = append_value[token]
    if layout == "HND":
        keys, values = keys.transpose(1, 2), values.transpose(1, 2)
        expected_k, expected_v = expected_k.transpose(1, 2), expected_v.transpose(1, 2)
    cache = torch.stack((keys, values), 1).to(npu) if packed else (keys.to(npu), values.to(npu))
    metadata = [torch.tensor(x, dtype=torch.int32, device=npu)
                for x in ([0, 1, 0], [1, 2, 5], [2, 0, 1], [0, 2, 3], [2, 3])]
    before = [x.clone() for x in metadata]
    batches, positions, indices, indptr, last = metadata
    assert flashinfer.append_paged_kv_cache(append_key.to(npu), append_value.to(npu), batches, positions, cache, indices, indptr, last, layout) is None
    actual_k, actual_v = (cache[:, 0], cache[:, 1]) if packed else cache
    torch.testing.assert_close(actual_k.cpu(), expected_k, atol=0, rtol=0)
    torch.testing.assert_close(actual_v.cpu(), expected_v, atol=0, rtol=0)
    assert all(torch.equal(a, b) for a, b in zip(metadata, before))


def test_reference_page_validation_precedes_mutation(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "reference")
    append = torch.ones(2, 1, 4)
    keys, values = torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4)
    ids = torch.tensor([0, 0], dtype=torch.int32)
    index, indptr, last = torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([2])
    with pytest.raises(ValueError, match="duplicate"):
        flashinfer.append_paged_kv_cache(append, append, ids, ids, (keys, values), index, indptr, last)
    assert not keys.any() and not values.any()
    flashinfer.append_paged_kv_cache(append, append, ids, torch.tensor([0, 1]), (keys, values), index, indptr, last)
    assert keys.eq(1).all() and values.eq(1).all()
