"""Public contract tests for attention.block_indexer_rearrange (arch22/950)."""

import math

import pytest
import torch

import flashinfer
from flashinfer.operators.attention.block_indexer_rearrange import cases
from flashinfer.operators.attention.block_indexer_rearrange.operator import (
    MAX_BLOCK_COUNT,
    MAX_BLOCK_SIZE,
    MAX_TOKENS,
    OPERATOR,
    infer,
    validate_values,
)
from flashinfer.operators.runtime import fake


def arguments(t1=7, n2=3, block_count=8, block_size=4, causal=False, device="cpu"):
    acc_q = [0, 3, 7]
    acc_kv = [0, 5, 9]
    return {
        "topk_indices": torch.zeros((t1, n2, block_count), dtype=torch.int32, device=device),
        "acc_seq_qlen": torch.tensor(acc_q, dtype=torch.int64, device=device),
        "acc_seq_kvlen": torch.tensor(acc_kv, dtype=torch.int64, device=device),
        "causal": causal,
        "block_size": block_size,
    }


def test_fake_derives_the_expanded_output_metadata():
    output = fake(OPERATOR.op, arguments(t1=7, n2=3, block_count=8, block_size=4))
    assert output.shape == (7, 3, 32)
    assert output.dtype == torch.int32
    assert output.device.type == "cpu"
    for block_size in (1, 2, 4, 8):
        output = fake(OPERATOR.op, arguments(block_size=block_size))
        assert output.shape == (7, 3, 8 * block_size)
    invocation = infer(arguments(causal=True), None)
    assert invocation.attributes == {"causal": True, "block_size": 4}
    assert invocation.inference.causal is True
    assert invocation.inference.compute_dtype == "int32"


def test_invalid_metadata_is_rejected_before_dispatch():
    for change in (
        {"block_size": 0},
        {"block_size": 9},
        {"block_size": 2.0},
        {"block_size": "4"},
        {"causal": 1},
        {"causal": "false"},
    ):
        values = arguments()
        values.update(change)
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("invalid attribute was accepted: " + repr(change))
    for change in (
        {"topk_indices": torch.empty((7, 3), dtype=torch.int32)},
        {"topk_indices": torch.empty((7, 3, 8), dtype=torch.int64)},
        {"acc_seq_qlen": torch.empty((3,), dtype=torch.int32)},
        {"acc_seq_kvlen": torch.empty((3, 1), dtype=torch.int64)},
        {"acc_seq_qlen": torch.empty((1,), dtype=torch.int64)},
    ):
        values = arguments()
        values.update(change)
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("invalid tensor metadata was accepted: " + repr(change))
    # blockCount beyond the 128 KiB broadcast buffer is rejected.
    try:
        fake(OPERATOR.op, arguments(block_count=MAX_BLOCK_COUNT + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("blockCount above 4096 was accepted")
    # T1 beyond the declared token range is rejected.
    try:
        beyond = arguments(block_count=1)
        beyond["topk_indices"] = torch.empty((MAX_TOKENS + 1, 1, 1), dtype=torch.int32)
        fake(OPERATOR.op, beyond)
    except ValueError:
        pass
    else:
        raise AssertionError("T1 above the declared range was accepted")


def test_validate_values_rejects_bad_prefixes_and_indices():
    values = arguments()
    values["acc_seq_qlen"] = torch.tensor([0, 4, 3], dtype=torch.int64)
    try:
        validate_values(values)
    except ValueError as error:
        assert "nondecreasing" in str(error)
    else:
        raise AssertionError("a non-monotonic prefix was accepted")

    values = arguments()
    values["acc_seq_kvlen"] = torch.tensor([1, 5, 9], dtype=torch.int64)
    try:
        validate_values(values)
    except ValueError as error:
        assert "starting at 0" in str(error)
    else:
        raise AssertionError("a nonzero prefix head was accepted")

    values = arguments()
    values["topk_indices"] = torch.full((7, 3, 8), -2, dtype=torch.int32)
    try:
        validate_values(values)
    except ValueError as error:
        assert "-1" in str(error)
    else:
        raise AssertionError("indices below -1 were accepted")

    values = arguments()
    values["acc_seq_qlen"] = torch.tensor([0, 3, 9], dtype=torch.int64)
    try:
        validate_values(values)
    except ValueError as error:
        assert "T1" in str(error)
    else:
        raise AssertionError("a prefix not ending at T1 was accepted")

    # Valid CPU inputs pass; non-CPU tensors are never read.
    validate_values(arguments())
    values = arguments(device="meta")
    values["topk_indices"] = torch.full((7, 3, 8), -2, dtype=torch.int32, device="meta")
    values["acc_seq_qlen"] = torch.tensor([5, 2, 1], dtype=torch.int64, device="meta")
    values["acc_seq_kvlen"] = torch.tensor([0, 1, 0], dtype=torch.int64, device="meta")
    validate_values(values)


def test_golden_matches_the_naive_position_oracle():
    """Self-consistency on a small ragged batch (valid count inside bc)."""
    tensors = cases.make_cpu_inputs(torch, dict(
        name="selfcheck", n2=2, cur_seqlen_q=(3, 4), cur_seqlen_kv=(5, 8),
        block_size=2, block_count=8, causal=True, seed=11))
    expected = cases.golden(torch, tensors)
    naive = cases.naive_reference(torch, tensors)
    assert torch.equal(expected, naive)

    # Hand-built ragged pair: token 0 has threshold=3 (one masked tail lane,
    # one unused block) and token 1 has threshold=4 (an exact block boundary).
    tensors = cases.make_cpu_inputs(torch, dict(
        name="hand", n2=1, cur_seqlen_q=(2,), cur_seqlen_kv=(4,),
        block_size=2, block_count=4, causal=True, seed=5))
    topk = tensors["topk_indices"]
    assert int(topk[0, 0, 1]) == 1 and int(topk[0, 0, 2]) == -1
    hand = cases.golden(torch, tensors)
    assert hand.shape == (2, 1, 8)
    assert hand[0, 0].tolist() == [0, 1, 2, -1, -1, -1, -1, -1]
    assert hand[1, 0].tolist() == [0, 1, 2, 3, -1, -1, -1, -1]


def test_case_parameters_cover_block_sizes_modes_and_heads():
    sizes = {case["block_size"] for case in cases.CASE_PARAMETERS}
    assert {1, 2, 4, 8} <= sizes
    assert max(sizes) <= MAX_BLOCK_SIZE
    modes = {case.get("causal", False) for case in cases.CASE_PARAMETERS}
    assert modes == {True, False}
    assert any(case.get("n2", 1) > 1 for case in cases.CASE_PARAMETERS)
    for case in cases.CASE_PARAMETERS:
        tensors = cases.make_cpu_inputs(torch, case)
        t1 = int(tensors["topk_indices"].shape[0])
        acc_q = tensors["acc_seq_qlen"].tolist()
        for t in range(t1):
            b_idx, s1_idx = cases._get_tnd_idx(acc_q, t)
            threshold = (tensors["cur_seqlen_kv"][b_idx]
                         - tensors["cur_seqlen_q"][b_idx] + 1 + s1_idx)
            valid = math.ceil(threshold / tensors["block_size"])
            assert valid <= tensors["block_count"], (case["name"], t)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
@pytest.mark.usefixtures("aot_runtime")
@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS, ids=lambda c: c["name"])
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    output = flashinfer.block_indexer_rearrange(*args, **options)
    torch.testing.assert_close(output.cpu(), expected[0], atol=0, rtol=0)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
@pytest.mark.usefixtures("aot_runtime")
def test_out_argument_reuses_supplied_buffers(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
    out = torch.empty_like(expected[0]).npu()
    returned = flashinfer.block_indexer_rearrange(*args, out=out, **options)
    returned = returned[0] if isinstance(returned, tuple) else returned
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(returned.cpu(), expected[0], atol=0, rtol=0)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend910B", "Ascend910C", "Ascend950")
@pytest.mark.usefixtures("aot_runtime")
def test_rejects_out_of_range_block_size(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, **cases.CASE_PARAMETERS[0])
    with pytest.raises(ValueError, match="block_size"):
        flashinfer.block_indexer_rearrange(*args, **dict(options, block_size=9))
    with pytest.raises(ValueError, match="block_size"):
        flashinfer.block_indexer_rearrange(*args, **dict(options, block_size=0))
