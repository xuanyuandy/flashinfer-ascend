"""Public QW contracts, FakeTensor metadata and native ABI declarations."""

import inspect
from dataclasses import replace

import pytest
import torch

import flashinfer
from flashinfer.ascendc import load_contract
from flashinfer.ascendc.attention.indexer_prologue_qw.implementation import IMPLEMENTATION as DENSE
from flashinfer.ascendc.attention.indexer_prologue_qw.implementation import (
    PREQUANTIZED_IMPLEMENTATION as PREQUANTIZED,
)
from flashinfer.operators.attention.indexer_prologue_qw.operator import OPERATOR
from flashinfer.operators.runtime import fake, signature
from flashinfer.runtime import GraphCapability, HardwareSignature, dispatch

DENSE_ID = "ascendc/attention/indexer_prologue_qw/native/op_package/Ascend910B"
PREQ_ID = "ascendc/attention/indexer_prologue_qw/prequantized/native/op_package/Ascend910B"


def dense_arguments(tokens=3, softmax_scale=128 ** -0.5):
    return dict(
        x=torch.zeros(tokens, 5120, dtype=torch.bfloat16),
        qr=torch.zeros(tokens, 1280, dtype=torch.bfloat16),
        wqb=torch.zeros(4096, 1280, dtype=torch.bfloat16),
        ww=torch.zeros(32, 5120, dtype=torch.bfloat16),
        rope_sin=torch.zeros(tokens, 64),
        rope_cos=torch.zeros(tokens, 64),
        softmax_scale=softmax_scale,
    )


def prequantized_arguments(tokens=3):
    args = dense_arguments(tokens)
    args["wqb"] = torch.zeros(1280, 4096, dtype=torch.int8)
    args["wqb_scale"] = torch.zeros(4096)
    args["prequantized"] = True
    return args


def test_public_exports_share_the_declared_interface():
    assert flashinfer.indexer_prologue_qw is flashinfer.attention.indexer_prologue_qw
    assert not hasattr(flashinfer, "indexer_prologue_qw_prequantized")
    assert inspect.signature(flashinfer.indexer_prologue_qw) == OPERATOR.public_signature


def test_fake_shapes_dtypes_and_dispatch():
    outputs = fake(OPERATOR.op, dense_arguments())
    assert [tuple(t.shape) for t in outputs] == [(3, 32, 128), (3, 32), (3, 32)]
    assert [t.dtype for t in outputs] == [torch.int8, torch.float16, torch.float16]
    outputs = fake(OPERATOR.op, prequantized_arguments())
    assert [tuple(t.shape) for t in outputs] == [(3, 32, 128), (3, 32), (3, 32)]
    request, _, _ = signature(OPERATOR.op, dense_arguments())
    assert dispatch(request, HardwareSignature("Ascend910B", 20)).kernel_id == DENSE_ID
    request, _, _ = signature(OPERATOR.op, prequantized_arguments())
    request = replace(request, physical_layouts=(
        "ND", "ND", "NZ", "ND", "ND", "ND", "ND", "ND", "ND", "ND"))
    assert dispatch(request, HardwareSignature("Ascend910B", 20)).kernel_id == PREQ_ID
    with pytest.raises(RuntimeError):
        dispatch(request, HardwareSignature("Ascend950", 32), kernel_id=DENSE_ID)


def test_native_abi_and_graph_capability_are_exact():
    dense = load_contract(DENSE_ID)[1]
    prequantized = load_contract(PREQ_ID)[1]
    expected = (
        "qr", "activation_q", "weight_q", "wqb", "weight_scale", "activation_scale",
        "qb", "x", "ww", "wb", "rope_sin", "rope_cos", "q", "q_scale", "w",
    )
    assert dense.tensor_parameter_names == expected
    assert prequantized.tensor_parameter_names == expected
    assert prequantized.tensor_parameters[2].physical_layouts == ("NZ",)
    assert prequantized.tensor_parameters[3].physical_layouts == ("NZ",)
    for implementation in (DENSE, PREQUANTIZED):
        assert implementation.bind().graph_capability is GraphCapability.NOT_CAPTURABLE


@pytest.mark.parametrize("change", [
    {"softmax_scale": float("nan")},
    {"softmax_scale": float("inf")},
    {"qr": torch.zeros(3, 64, dtype=torch.bfloat16)},
    {"wqb": torch.zeros(4096, 1280, dtype=torch.float16)},
    {"rope_sin": torch.zeros(3, 64, dtype=torch.bfloat16)},
])
def test_invalid_dense_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        fake(OPERATOR.op, dict(dense_arguments(), **change))


@pytest.mark.parametrize("change", [
    {"wqb": torch.zeros(1280, 4096, dtype=torch.float16)},
    {"wqb_scale": torch.zeros(4096, dtype=torch.float16)},
    {"wqb": torch.zeros(1281, 4096, dtype=torch.int8)},
])
def test_invalid_prequantized_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        fake(OPERATOR.op, dict(prequantized_arguments(), **change))


def test_zero_tokens_are_rejected_for_qw():
    with pytest.raises(ValueError):
        fake(OPERATOR.op, dense_arguments(tokens=0))


def test_device_user_workspace_and_public_token_length_proofs():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2]
              / "csrc/kernels/attention/indexer_prologue_qw/op_kernel/indexer_prologue_qw.cpp").read_text()
    assert "reinterpret_cast<__gm__ int32_t*>(GetUserWorkspace(workspace))" in source
    normalized = " ".join(source.split())
    for expression in (
        "static_cast<uint64_t>(tilingData->get_totalInGroup()) * tilingData->get_k()",
        "static_cast<uint64_t>(tilingData->get_totalInGroup()) * 5120",
        "static_cast<uint64_t>(tilingData->get_totalInGroup()) * 32",
    ):
        assert expression in normalized

    max_tokens = 1048576
    assert max_tokens * 1280 == 1342177280
    assert max_tokens * 5120 == 5368709120 > 2 ** 32 - 1
    assert max_tokens * 32 == 33554432
    # uint64_t covers every public-T element count with ample headroom.
    assert max(max_tokens * 1280, max_tokens * 5120, max_tokens * 32) < 2 ** 64
