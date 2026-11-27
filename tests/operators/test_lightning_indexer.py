"""Public numerics for the experimental arch22 lightning indexer."""

import pytest

import flashinfer
from flashinfer.operators.sparse_indexer.lightning_indexer import cases

pytestmark = [
    pytest.mark.real_compile,
    pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
    pytest.mark.usefixtures("aot_runtime"),
]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    indices, values = flashinfer.lightning_indexer(*args, **options)
    (atol_indices, rtol_indices), (atol_values, rtol_values) = cases.tolerances(configuration)
    torch.testing.assert_close(indices.cpu().double(), expected[0].double(),
                               atol=atol_indices, rtol=rtol_indices)
    torch.testing.assert_close(values.cpu().double(), expected[1].double(),
                               atol=atol_values, rtol=rtol_values)


def test_out_argument_reuses_supplied_buffers(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
    out = tuple(torch.empty_like(tensor).npu() for tensor in expected)
    returned = flashinfer.lightning_indexer(*args, out=out, **options)
    assert tuple(tensor.data_ptr() for tensor in returned) == tuple(tensor.data_ptr() for tensor in out)


def test_non_positive_token_count_is_rejected(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
    with pytest.raises(ValueError, match="sparse_count"):
        flashinfer.lightning_indexer(*args, **dict(options, sparse_count=100))
    with pytest.raises(ValueError, match="sparse_count"):
        flashinfer.lightning_indexer(*args, **dict(options, sparse_count=8192))


def test_unsupported_q_block_len_is_rejected(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
    with pytest.raises(ValueError, match="q_block_len"):
        flashinfer.lightning_indexer(*args, **dict(options, q_block_len=2))


def test_layout_mismatch_is_rejected(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
    with pytest.raises(ValueError, match="layout"):
        flashinfer.lightning_indexer(*args, **dict(options, layout_query="BSND", layout_key="TND"))
