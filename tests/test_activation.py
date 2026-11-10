import pytest
import torch
import torch.nn.functional as functional

import flashinfer


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [6, 128, 4096])
def test_activation_cann_numerics_out_and_pdl(monkeypatch, name, dtype, width):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("requires verified Ascend950")
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "cann")
    torch.manual_seed(42)
    source = torch.randn(2, 3, width, dtype=dtype)
    a, b = source.float().chunk(2, -1)
    expected = ((functional.silu(a) if name == "silu_and_mul" else functional.gelu(
        a, approximate="tanh" if name == "gelu_tanh_and_mul" else "none")) * b).to(dtype)
    source = source.npu()
    out = torch.empty(expected.shape, dtype=dtype, device=source.device)
    result = getattr(flashinfer, name)(source, out, enable_pdl=True)
    assert result is out
    torch.testing.assert_close(out.cpu(), expected, rtol=0.015 if dtype == torch.bfloat16 else 0.002,
                               atol=0.015 if dtype == torch.bfloat16 else 0.002)


def test_reference_and_invalid_activation(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "reference")
    source = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert flashinfer.gelu_and_mul(source).shape == (1, 2)
    with pytest.raises(ValueError, match="even"):
        flashinfer.silu_and_mul(torch.ones(1, 3))
    with pytest.raises(ValueError, match="out"):
        flashinfer.gelu_and_mul(source, torch.empty(1, 4))
