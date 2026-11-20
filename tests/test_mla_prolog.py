"""CPU-only public contract tests for attention.mla_prolog (A3)."""

import inspect

import torch

from flashinfer.operators.attention.mla_prolog import cases
from flashinfer.operators.attention.mla_prolog.operator import (
    CACHE_PAGES, CACHE_ROWS, HEADS, HEAD_DIM, HIDDEN, KV_LORA, OPERATOR,
    Q_LORA, Q_OUT, RMS_NORM_EPS_DEFAULT, ROPE_DIM, torch_declarations,
    validate_values,
)
from flashinfer.operators.runtime import fake


def meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def arguments(t=1, rms_norm_eps=RMS_NORM_EPS_DEFAULT):
    return {
        "x": meta((t, HIDDEN), torch.bfloat16),
        "w_qa": meta((Q_LORA, HIDDEN), torch.bfloat16),
        "gamma_qa": meta((Q_LORA,), torch.bfloat16),
        "w_qb": meta((Q_OUT // 32, Q_LORA // 16, 16, 32), torch.int8),
        "w_qb_scale": meta((Q_OUT,), torch.bfloat16),
        "w_kv": meta((KV_LORA, HIDDEN), torch.bfloat16),
        "gamma_kv": meta((KV_LORA,), torch.bfloat16),
        "rope_cos": meta((t, 1, 1, ROPE_DIM), torch.float32),
        "rope_sin": meta((t, 1, 1, ROPE_DIM), torch.float32),
        "slot_indices": meta((t, 2), torch.int32),
        "kv_cache": meta((CACHE_PAGES, CACHE_ROWS, KV_LORA), torch.bfloat16),
        "rms_norm_eps": rms_norm_eps,
    }


def test_public_entry_and_metadata_only_fake():
    import flashinfer.attention.mla_prolog as module

    assert OPERATOR.public_entry == "flashinfer.attention.mla_prolog:mla_prolog"
    assert OPERATOR.output_names == ("q_output", "qr")
    assert inspect.signature(module.mla_prolog) == OPERATOR.public_signature
    parameters = inspect.signature(module.mla_prolog).parameters
    assert parameters["rms_norm_eps"].default == RMS_NORM_EPS_DEFAULT
    assert parameters["qr_out"].default is None
    outputs = fake(OPERATOR.op, arguments(3))
    assert isinstance(outputs, tuple) and len(outputs) == 2
    q_output, qr = outputs
    assert q_output.shape == (3, HEADS, HEAD_DIM)
    assert q_output.dtype == torch.bfloat16
    assert q_output.device.type == "meta"
    assert qr.shape == (3, Q_LORA)
    assert qr.dtype == torch.bfloat16
    assert qr.device.type == "meta"


def test_rms_norm_eps_is_validated_before_dispatch():
    assert fake(OPERATOR.op, arguments(1, rms_norm_eps=1.0e-20))[1].shape == (1, Q_LORA)
    for value in (0, -1.0e-6, float("nan"), float("inf"), True):
        values = arguments()
        values["rms_norm_eps"] = value
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("invalid rms_norm_eps was accepted: " + repr(value))


def test_torch_declarations_declare_kv_cache_mutation():
    declarations = torch_declarations()
    assert [item.name for item in declarations] == [
        "flashinfer::mla_prolog", "flashinfer::mla_prolog_out"]
    for declaration in declarations:
        schema = torch._C.parse_schema(declaration.name + declaration.schema)
        mutated = {
            arg.name for arg in schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        }
        assert mutated == set(declaration.mutates_args)
    assert declarations[0].mutates_args == ("kv_cache",)
    assert declarations[1].mutates_args == ("kv_cache", "q_output", "qr")
    functional = torch._C.parse_schema(
        declarations[0].name + declarations[0].schema)
    assert [str(value.type) for value in functional.returns] == [
        "Tensor", "Tensor"]
    assert functional.arguments[-3].name == "rms_norm_eps"
    assert functional.arguments[-3].default_value == 1e-6


def test_invalid_metadata_is_rejected_before_dispatch():
    for change in (
        {"x": meta((0, HIDDEN), torch.bfloat16)},
        {"w_qb": meta((Q_LORA, Q_OUT), torch.int8)},
        {"kv_cache": meta((1, CACHE_ROWS, KV_LORA), torch.bfloat16)},
        {"rope_cos": meta((1, 1, ROPE_DIM), torch.float32)},
    ):
        values = arguments()
        values.update(change)
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("invalid metadata was accepted: " + repr(change))


def test_validate_values_rejects_duplicate_slots_and_bad_rope():
    values = arguments(2)
    values.update(
        slot_indices=torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
        rope_cos=torch.ones(2, 1, 1, ROPE_DIM),
        rope_sin=torch.zeros(2, 1, 1, ROPE_DIM),
        w_qb_scale=torch.ones(Q_OUT, dtype=torch.bfloat16),
        gamma_qa=torch.ones(Q_LORA, dtype=torch.bfloat16),
        gamma_kv=torch.ones(KV_LORA, dtype=torch.bfloat16),
    )
    try:
        validate_values(values)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate slot_indices were accepted")

    values["slot_indices"] = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    values["rope_cos"] = values["rope_cos"].clone()
    values["rope_cos"][0, 0, 0, 1] = 2.0
    try:
        validate_values(values)
    except ValueError as error:
        assert "rope_cos" in str(error)
    else:
        raise AssertionError("unequal adjacent rope_cos values were accepted")


def test_nz_pack_roundtrip_and_partial_rope_primitive():
    logical = torch.arange(16 * 32, dtype=torch.int16).to(torch.int8).reshape(16, 32)
    packed = cases.pack_weight_nz(torch, logical)
    assert packed.shape == (1, 1, 16, 32)
    assert torch.equal(cases.unpack_weight_nz(torch, packed, 16, 32), logical)

    value = torch.zeros((1, HEAD_DIM))
    value[0, 448:450] = torch.tensor([1.0, 2.0])
    cos = torch.ones((1, 1, 1, ROPE_DIM))
    sin = torch.zeros_like(cos)
    cos[..., :2] = 0.0
    sin[..., :2] = 1.0
    rotated = cases._partial_rope_bf16(torch, value, cos, sin)
    assert torch.equal(rotated[0, 448:450], torch.tensor([-2.0, 1.0]))


def test_qr_golden_is_the_bf16_rms_norm_row():
    torch.manual_seed(0)
    x = (torch.randn(4, HIDDEN) * 0.05).to(torch.bfloat16)
    w_qa = (torch.randn(Q_LORA, HIDDEN) * 0.02).to(torch.bfloat16)
    gamma_qa = (torch.rand(Q_LORA) * 0.5 + 0.75).to(torch.bfloat16)
    w_qb = torch.randint(-4, 5, (Q_LORA, Q_OUT), dtype=torch.int8)
    w_qb_scale = (torch.rand(Q_OUT) * 0.004 + 0.002).to(torch.bfloat16)
    w_kv = (torch.randn(KV_LORA, HIDDEN) * 0.02).to(torch.bfloat16)
    gamma_kv = (torch.rand(KV_LORA) * 0.5 + 0.75).to(torch.bfloat16)
    angles = torch.rand(4, 1, 1, ROPE_DIM // 2) * 0.5
    rope_cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    rope_sin = torch.sin(angles).repeat_interleave(2, dim=-1)

    q_output, qr, kv_updates = cases.golden(
        torch, x, w_qa, gamma_qa, cases.pack_weight_nz(torch, w_qb),
        w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin)
    assert qr.shape == (4, Q_LORA) and q_output.shape == (4, HEADS, HEAD_DIM)
    assert kv_updates.shape == (4, KV_LORA)

    qa = cases._matmul_bf16(
        torch, cases.bf16_round(torch, x.float()),
        cases.bf16_round(torch, w_qa.float()))
    expected_qr = cases._rms_norm_bf16(torch, qa, gamma_qa, RMS_NORM_EPS_DEFAULT)
    assert torch.equal(qr, expected_qr)

    _, qr_tight, _ = cases.golden(
        torch, x, w_qa, gamma_qa, cases.pack_weight_nz(torch, w_qb),
        w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin,
        rms_norm_eps=1.0e-20)
    assert qr_tight.shape == qr.shape
    assert not torch.equal(qr_tight, qr)


def test_case_tolerances_match_outputs():
    assert len(cases.CASE_PARAMETERS) >= 1
    assert len(cases.TOLERANCES) == len(OPERATOR.output_names) == 2
    assert any("rms_norm_eps" in case for case in cases.CASE_PARAMETERS)
