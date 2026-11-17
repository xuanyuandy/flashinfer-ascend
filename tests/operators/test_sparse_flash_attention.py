"""Public sparse attention accuracy: sample, call, compare."""

import pytest

import flashinfer
from flashinfer.operators.attention.sparse_flash_attention import cases

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = flashinfer.sparse_flash_attention(*args, **options)
    for value, reference, (atol, rtol) in zip(actual, expected, cases.TOLERANCES):
        torch.testing.assert_close(value.cpu().double(), reference.double(), atol=atol, rtol=rtol)
