"""Public GEMM numerical comparisons; setup and compilation live in fixtures."""

import pytest

import flashinfer
from flashinfer.operators.gemm.matmul import cases

pytestmark = pytest.mark.requires_soc("Ascend950")


@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 64, 48), (80, 48, 32),
                                  (128, 128, 128), (256, 256, 256), (256, 16, 256), (16, 256, 16)])
def test_matmul_accuracy(npu_runtime, matmul_backend, shape):
    torch, torch_npu = npu_runtime
    m, n, k = shape
    args, options, expected = cases.sample(torch, torch_npu, m=m, k=k, n=n)
    actual = flashinfer.matmul(*args, **options, backend=matmul_backend)
    (atol, rtol), = cases.TOLERANCES
    torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
