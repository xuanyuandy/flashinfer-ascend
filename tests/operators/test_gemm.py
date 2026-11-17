"""Public GEMM numerical comparisons; setup and compilation live in fixtures."""

import pytest

import flashinfer

pytestmark = pytest.mark.requires_soc("Ascend950")


@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 64, 48), (80, 48, 32),
                                  (128, 128, 128), (256, 256, 256), (256, 16, 256), (16, 256, 16)])
def test_matmul_accuracy(npu_runtime, matmul_backend, shape):
    torch, _ = npu_runtime
    torch.manual_seed(20261109)
    m, n, k = shape
    a, b = (torch.randn(m, k) * .25).bfloat16(), (torch.randn(k, n) * .25).bfloat16()
    actual = flashinfer.matmul(a.npu(), b.npu(), backend=matmul_backend)
    expected = a.double() @ b.double()
    torch.testing.assert_close(actual.cpu().double(), expected, atol=.008, rtol=.008)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
@pytest.mark.usefixtures("aot_runtime")
def test_matmul_bias_silu_accuracy(npu_runtime):
    torch, _ = npu_runtime
    torch.manual_seed(20261026)
    a, b = (torch.randn(16, 48) * .25).bfloat16(), (torch.randn(48, 32) * .25).bfloat16()
    bias = (torch.randn(32) * .1).bfloat16()
    actual = flashinfer.matmul_bias_silu(a.npu(), b.npu(), bias.npu())
    expected = torch.nn.functional.silu(a.double() @ b.double() + bias.double())
    torch.testing.assert_close(actual.cpu().double(), expected, atol=.008, rtol=.008)
