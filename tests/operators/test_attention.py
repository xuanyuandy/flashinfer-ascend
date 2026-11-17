"""Public CATLASS prefill accuracy with an independent CPU calculation."""

import pytest

import flashinfer

pytestmark = [pytest.mark.real_compile, pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("lengths", [(128, 128), (128, 256), (256, 512)])
def test_catlass_prefill_accuracy(npu_runtime, dtype, lengths):
    torch, _ = npu_runtime
    torch.manual_seed(20261026)
    qlen, klen = lengths
    q, k, v = ((torch.rand(shape) * 2 - 1).to(getattr(torch, dtype)) for shape in
               [(qlen, 2, 128), (klen, 2, 128), (klen, 2, 128)])
    actual = flashinfer.single_prefill_with_kv_cache_catlass(q.npu(), k.npu(), v.npu())
    query, key, value = (tensor.double().transpose(0, 1) for tensor in (q, k, v))
    expected = (torch.softmax((query @ key.transpose(1, 2)) / 128 ** .5, -1) @ value).transpose(0, 1)
    atol, rtol = (.001, .001) if dtype == "float16" else (.004, .008)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
