"""Public grouped GEMM accuracy for the declared quantization cases."""

import pytest

import flashinfer
from flashinfer.operators.gemm.grouped_matmul_swiglu_quant import cases

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    # CANN format IDs: ND=2, FRACTAL_NZ=29. Check that the requested case is exercised.
    expected_format = 29 if configuration.get("layout", "NZ") == "NZ" else 2
    assert torch_npu.get_npu_format(args[1]) == expected_format
    actual = flashinfer.grouped_matmul_swiglu_quant(*args, **options)
    for value, reference, (atol, rtol) in zip(actual, expected, cases.TOLERANCES):
        torch.testing.assert_close(value.cpu().double(), reference.double(), atol=atol, rtol=rtol)
