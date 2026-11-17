"""Public operator numerical comparisons."""


import pytest
import torch
import flashinfer.sampling as sampling
from flashinfer._sampling import OP_NAMES

@pytest.fixture
def npu(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(torch.npu.current_device()):
        pytest.skip("requires verified Ascend950")
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "cann")
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.requires_soc("Ascend950")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cann_filters_and_all_sampling_entrypoints(npu, dtype):
    probs = torch.tensor([[0.5, 0.25, 0.25, 0]], dtype=dtype, device=npu)
    torch.testing.assert_close(sampling.top_k_renorm_probs(probs, 2).cpu(), probs.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(sampling.top_p_renorm_probs(probs, 0.6).cpu(), probs.cpu(), rtol=0, atol=0)
    assert torch.isneginf(sampling.top_k_mask_logits(probs, 1).cpu()[0, 1:]).all()
    one_hot = torch.tensor([[0, 1, 0, 0]], dtype=dtype, device=npu)
    for name, args in (
        ("sampling_from_probs", (one_hot,)),
        ("sampling_from_logits", (torch.tensor([[-100, 100, -100, -100]], dtype=dtype, device=npu),)),
        ("top_k_sampling_from_probs", (one_hot, 1)),
        ("top_p_sampling_from_probs", (one_hot, 0.5)),
        ("top_k_top_p_sampling_from_probs", (one_hot, 2, 0.5)),
        ("min_p_sampling_from_probs", (one_hot, 0.5)),
    ):
        assert getattr(sampling, name)(*args).cpu().tolist() == [1]
    draft = one_hot.unsqueeze(1)
    output, a, e = sampling.chain_speculative_sampling(draft, torch.tensor([[1]], device=npu), draft.repeat(1, 2, 1))
    assert output.cpu().tolist() == [[1, 1]]
    assert a.cpu().item() == e.cpu().item() == 1
    assert len(OP_NAMES) == 10


@pytest.mark.requires_soc("Ascend950")
def test_cann_distribution_and_reproducibility(npu):
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=npu).repeat(6000, 1)
    generator = torch.Generator(device=npu).manual_seed(123)
    first = sampling.sampling_from_probs(probs, generator=generator)
    second = sampling.sampling_from_probs(probs, generator=generator.manual_seed(123))
    assert torch.equal(first, second)
    frequencies = torch.bincount(first.cpu().long(), minlength=4) / probs.shape[0]
    torch.testing.assert_close(frequencies, probs[0].cpu(), atol=0.03, rtol=0)
