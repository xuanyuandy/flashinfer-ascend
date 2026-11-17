import inspect

import pytest
import torch

import flashinfer.sampling as sampling
from flashinfer._sampling import OP_NAMES


@pytest.fixture(autouse=True)
def reference(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NPU_BACKEND", "reference")


def test_signatures_preserve_upstream_positions_and_defaults():
    suffix = ["indices", "deterministic", "generator", "check_nan", "seed", "offset", "return_valid"]
    for name, threshold in (("top_k_sampling_from_probs", "top_k"),
                            ("top_p_sampling_from_probs", "top_p"),
                            ("min_p_sampling_from_probs", "min_p")):
        params = inspect.signature(getattr(sampling, name)).parameters
        assert list(params) == ["probs", threshold] + suffix
        assert params["deterministic"].default is True
        assert params["return_valid"].default is False
    assert list(inspect.signature(sampling.top_k_top_p_sampling_from_probs).parameters) == [
        "probs", "top_k", "top_p", "indices", "filter_apply_order", "deterministic",
        "generator", "check_nan", "seed", "offset", "return_valid",
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_renormalization_preserves_ties_and_input_dtype(dtype):
    probs = torch.tensor([[0.5, 0.25, 0.25, 0], [0.25] * 4], dtype=dtype)
    assert torch.equal(sampling.top_k_renorm_probs(probs, 2), probs)
    assert torch.equal(sampling.top_k_renorm_probs(probs[1:], 1), probs[1:])
    assert torch.equal(sampling.top_p_renorm_probs(probs, 0.6), probs)
    expected = torch.tensor([[1, 0, 0, 0]], dtype=dtype)
    assert torch.equal(sampling.top_p_renorm_probs(probs[:1], 0.5), expected)
    assert sampling.top_k_mask_logits(probs, 1).dtype == dtype
    assert torch.isneginf(sampling.top_k_mask_logits(probs[:1], 1)[0, 1:]).all()


def test_indices_thresholds_and_invalid_rows():
    probs = torch.tensor([[1.0, 0, 0], [0, 0.4, 0.6], [0, 0, 0]])
    indices = torch.tensor([1, 0, 1, 2], dtype=torch.int64)
    samples, valid = sampling.top_k_sampling_from_probs(probs, torch.tensor([1, 1, 2]), indices, return_valid=True)
    assert samples.tolist() == [2, 0, 2, 0]
    assert valid.tolist() == [True, True, True, False]
    assert samples.dtype == indices.dtype
    assert sampling.sampling_from_probs(probs[:0]).shape == (0,)


def test_orders_have_distinct_filtering_semantics():
    probs = torch.tensor([[0.45, 0.25, 0.2, 0.1]]).repeat(1000, 1)
    first = sampling.top_k_top_p_sampling_from_probs(probs, 2, 0.6, filter_apply_order="top_k_first")
    joint = sampling.top_k_top_p_sampling_from_probs(probs, 2, 0.6, filter_apply_order="joint")
    assert torch.equal(first, torch.zeros_like(first))
    assert set(joint.tolist()) == {0, 1}


def test_sampling_distribution_and_local_reproducibility():
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).repeat(8000, 1)
    generator = torch.Generator().manual_seed(71)
    first = sampling.sampling_from_probs(probs, generator=generator)
    second = sampling.sampling_from_probs(probs, generator=generator.manual_seed(71))
    assert torch.equal(first, second)
    frequencies = torch.bincount(first.long(), minlength=4) / probs.shape[0]
    torch.testing.assert_close(frequencies, probs[0], atol=0.025, rtol=0)
    assert sampling.sampling_from_logits(torch.tensor([[-10.0, 10.0]])).item() == 1


def test_min_p_boundary_keeps_equal_maxima():
    probs = torch.tensor([[0.4, 0.4, 0.2]]).repeat(512, 1)
    assert set(sampling.min_p_sampling_from_probs(probs, 1.0).tolist()) == {0, 1}


def test_speculative_metrics_accumulate_and_stop_at_first_rejection():
    draft = torch.tensor([[[1.0, 0], [1.0, 0], [1.0, 0]]])
    target = torch.tensor([[[0.0, 1], [1.0, 0], [1.0, 0], [0.0, 1]]])
    ids = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    accepted, emitted = torch.tensor([10], dtype=torch.int32), torch.tensor([20], dtype=torch.int32)
    output, a, e = sampling.chain_speculative_sampling(
        draft_probs=draft, draft_token_ids=ids, target_probs=target,
        maybe_output_accepted_token_num=accepted,
        maybe_output_emitted_draft_token_num=emitted,
    )
    assert a is accepted and e is emitted
    assert output.tolist() == [[1, -1, -1, -1]]
    assert accepted.item() == 12
    assert emitted.item() == 20
    output, a, e = sampling.chain_speculative_sampling(draft, ids, torch.cat((draft, target[:, -1:]), dim=1))
    assert output.tolist() == [[0, 0, 0, 1]]
    assert (a.item(), e.item()) == (3, 3)


def test_invalid_controls_fail_explicitly():
    probs = torch.tensor([[0.2, 0.8]])
    with pytest.raises(NotImplementedError, match="Philox"):
        sampling.sampling_from_probs(probs, seed=1, offset=0)
    with pytest.raises(ValueError, match="NaN"):
        sampling.sampling_from_probs(torch.tensor([[float("nan"), 1.0]]), check_nan=True)
    with pytest.raises(ValueError, match="top_k"):
        sampling.top_k_renorm_probs(probs, 0)
    with pytest.raises(ValueError, match="filter_apply_order"):
        sampling.top_k_top_p_sampling_from_probs(probs, 1, 0.5, filter_apply_order="reverse")
