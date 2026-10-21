import pytest
import torch

import flashinfer
from flashinfer.attention import ReferenceTensor


def index(values):
    return ReferenceTensor.from_nested(values, dtype="int32")


@pytest.mark.parametrize("enable_pdl", [None, False, True])
@pytest.mark.parametrize("mode", ["paged", "ragged", "decode"])
def test_attention_pdl_is_a_hardware_noop(mode, enable_pdl):
    workspace = ReferenceTensor.zeros((128,), dtype="uint8")
    q = ReferenceTensor.from_nested([[[0.0]]])
    k = ReferenceTensor.from_nested([[[0.0]], [[0.0]]])
    v = ReferenceTensor.from_nested([[[4.0]], [[8.0]]])
    if mode == "ragged":
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, backend="reference")
        wrapper.plan(index([0, 1]), index([0, 2]), 1, 1, 1, q_data_type="float32")
        output = wrapper.run(q, k, v, enable_pdl=enable_pdl)
    else:
        cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper if mode == "paged" else flashinfer.BatchDecodeWithPagedKVCacheWrapper
        wrapper = cls(workspace, backend="reference")
        args = (index([0, 1]), index([0]), index([2]), 1, 1, 1, 2)
        if mode == "paged":
            args = (index([0, 1]), *args)
        wrapper.plan(*args, q_data_type="float32")
        kp = ReferenceTensor((1, 2, 1, 1), k.data, k.dtype)
        vp = ReferenceTensor((1, 2, 1, 1), v.data, v.dtype)
        output = wrapper.run(q, (kp, vp), enable_pdl=enable_pdl)
    assert output.data == (6.0,)


@pytest.mark.parametrize("enable_pdl", [None, False, True])
def test_norm_and_activation_share_pdl_semantics(monkeypatch, enable_pdl):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "reference")
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weight = torch.ones(4)
    actual = flashinfer.rmsnorm(x, weight, enable_pdl=enable_pdl)
    expected = flashinfer.rmsnorm(x, weight, enable_pdl=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual = flashinfer.silu_and_mul(x, enable_pdl=enable_pdl)
    expected = flashinfer.silu_and_mul(x, enable_pdl=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
