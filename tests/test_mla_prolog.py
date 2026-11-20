"""CPU-only public contract tests for attention.mla_prolog (A3)."""

import inspect

import torch

from flashinfer.operators.attention.mla_prolog import cases
from flashinfer.operators.attention.mla_prolog.operator import (
    CACHE_PAGES, CACHE_ROWS, DEFAULT_HEAD_NUM, HEAD_DIM, HIDDEN, KV_LORA,
    MAX_HEAD_NUM, OPERATOR, Q_LORA, RMS_NORM_EPS_DEFAULT, ROPE_DIM,
    UPDATE_KV_CACHE_DEFAULT, infer, q_out_dim, torch_declarations,
    validate_values, w_qb_nz_shape,
)
from flashinfer.operators.runtime import fake

HEAD_NUM_CASES = (1, 16, 31, 32, 63, 64)


def meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def arguments(t=1, rms_norm_eps=RMS_NORM_EPS_DEFAULT,
              q_head_num=DEFAULT_HEAD_NUM,
              update_kv_cache=UPDATE_KV_CACHE_DEFAULT):
    return {
        "x": meta((t, HIDDEN), torch.bfloat16),
        "w_qa": meta((Q_LORA, HIDDEN), torch.bfloat16),
        "gamma_qa": meta((Q_LORA,), torch.bfloat16),
        "w_qb": meta(w_qb_nz_shape(q_head_num), torch.int8),
        "w_qb_scale": meta((q_out_dim(q_head_num),), torch.bfloat16),
        "w_kv": meta((KV_LORA, HIDDEN), torch.bfloat16),
        "gamma_kv": meta((KV_LORA,), torch.bfloat16),
        "rope_cos": meta((t, 1, 1, ROPE_DIM), torch.float32),
        "rope_sin": meta((t, 1, 1, ROPE_DIM), torch.float32),
        "slot_indices": meta((t, 2), torch.int32),
        "kv_cache": meta((CACHE_PAGES, CACHE_ROWS, KV_LORA), torch.bfloat16),
        "rms_norm_eps": rms_norm_eps,
        "q_head_num": q_head_num,
        "update_kv_cache": update_kv_cache,
    }


def test_public_entry_and_metadata_only_fake():
    import flashinfer.attention.mla_prolog as module

    assert OPERATOR.public_entry == "flashinfer.attention.mla_prolog:mla_prolog"
    assert OPERATOR.output_names == ("q_output", "qr", "kv")
    assert inspect.signature(module.mla_prolog) == OPERATOR.public_signature
    parameters = inspect.signature(module.mla_prolog).parameters
    assert parameters["rms_norm_eps"].default == RMS_NORM_EPS_DEFAULT
    assert parameters["q_head_num"].default == DEFAULT_HEAD_NUM == 64
    assert parameters["update_kv_cache"].default is UPDATE_KV_CACHE_DEFAULT
    assert parameters["update_kv_cache"].default is True
    assert parameters["qr_out"].default is None
    assert parameters["kv_out"].default is None
    outputs = fake(OPERATOR.op, arguments(3))
    assert isinstance(outputs, tuple) and len(outputs) == 3
    q_output, qr, kv = outputs
    assert q_output.shape == (3, DEFAULT_HEAD_NUM, HEAD_DIM)
    assert q_output.dtype == torch.bfloat16
    assert q_output.device.type == "meta"
    assert qr.shape == (3, Q_LORA)
    assert qr.dtype == torch.bfloat16
    assert qr.device.type == "meta"
    # kv is declared in both modes; the fused kernel leaves it unwritten.
    assert kv.shape == (3, KV_LORA)
    assert kv.dtype == torch.bfloat16
    assert kv.device.type == "meta"


def test_q_head_num_drives_only_the_q_output_shape():
    for q_head_num in HEAD_NUM_CASES:
        q_output, qr, kv = fake(OPERATOR.op, arguments(2, q_head_num=q_head_num))
        assert q_output.shape == (2, q_head_num, HEAD_DIM)
        assert q_output.dtype == torch.bfloat16
        assert q_output.device.type == "meta"
        # qr is Q-LoRA normalization and is never sharded by head count.
        assert qr.shape == (2, Q_LORA)
        assert qr.dtype == torch.bfloat16
        # kv is never sharded by head count either.
        assert kv.shape == (2, KV_LORA)
        assert kv.dtype == torch.bfloat16
    assert MAX_HEAD_NUM == 64


def test_update_kv_cache_selects_the_public_return_contract():
    for q_head_num in HEAD_NUM_CASES:
        invocation = infer(arguments(2, q_head_num=q_head_num,
                                     update_kv_cache=False), None)
        assert [spec.shape for spec in invocation.outputs] == [
            (2, q_head_num, HEAD_DIM), (2, Q_LORA), (2, KV_LORA)]
        assert invocation.attributes["update_kv_cache"] is False
    fused = infer(arguments(2), None)
    assert fused.attributes["update_kv_cache"] is True
    assert len(fused.outputs) == len(OPERATOR.output_names) == 3
    # The public facade returns the pair in fused mode and the triple in
    # diagnostic mode; cases encodes the same per-case count.
    assert cases.output_count({}) == 2
    assert cases.output_count({"update_kv_cache": True}) == 2
    assert cases.output_count({"update_kv_cache": False}) == 3


def test_public_result_is_the_pair_in_fused_mode_and_triple_otherwise():
    import flashinfer.attention.mla_prolog as module

    outputs = ("q_output", "qr", "kv")
    assert module._public_result(outputs, True) == ("q_output", "qr")
    assert module._public_result(outputs, False) == (
        "q_output", "qr", "kv")
    for value in (0, 1, "true", None):
        try:
            module._public_result(outputs, value)
        except ValueError:
            continue
        raise AssertionError("non-bool update_kv_cache was accepted: " + repr(value))
    try:
        module._public_result(outputs[:2], True)
    except ValueError:
        pass
    else:
        raise AssertionError("a short native output list was accepted")


def test_invalid_update_kv_cache_is_rejected_before_dispatch():
    for value in (0, 1, "false", None, 1.0):
        values = arguments(1)
        values["update_kv_cache"] = value
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("invalid update_kv_cache was accepted: " + repr(value))


def test_validate_values_ignores_slots_only_when_cache_is_disabled():
    def values(update):
        result = arguments(2, update_kv_cache=update)
        result.update(
            # Duplicate addresses that the fused scatter rejects.
            slot_indices=torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
            rope_cos=torch.ones(2, 1, 1, ROPE_DIM),
            rope_sin=torch.zeros(2, 1, 1, ROPE_DIM),
            w_qb_scale=torch.ones(q_out_dim(), dtype=torch.bfloat16),
            gamma_qa=torch.ones(Q_LORA, dtype=torch.bfloat16),
            gamma_kv=torch.ones(KV_LORA, dtype=torch.bfloat16),
        )
        return result

    try:
        validate_values(values(True))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate slot_indices were accepted in fused mode")
    # slot_indices is ignored (never read) when the cache is not updated.
    validate_values(values(False))


def test_invalid_q_head_num_is_rejected_before_dispatch():
    for value in (0, 65, -1, True, 1.0, "64", None):
        values = arguments(1)
        values["q_head_num"] = value
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("invalid q_head_num was accepted: " + repr(value))


def test_w_qb_and_scale_shapes_must_match_q_head_num():
    for change in (
        {"w_qb": meta(w_qb_nz_shape(DEFAULT_HEAD_NUM), torch.int8)},
        {"w_qb_scale": meta((q_out_dim(DEFAULT_HEAD_NUM),), torch.bfloat16)},
    ):
        values = arguments(1, q_head_num=16)
        values.update(change)
        try:
            fake(OPERATOR.op, values)
        except ValueError:
            continue
        raise AssertionError("mismatched q_head_num weights accepted: " + repr(change))


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
    assert declarations[1].mutates_args == (
        "kv_cache", "q_output", "qr", "kv")
    functional = torch._C.parse_schema(
        declarations[0].name + declarations[0].schema)
    # The ACLNN output list is static: all three outputs are declared and the
    # public facade truncates the fused mode to the legacy pair.
    assert [str(value.type) for value in functional.returns] == [
        "Tensor", "Tensor", "Tensor"]
    assert functional.arguments[-5].name == "rms_norm_eps"
    assert functional.arguments[-5].default_value == 1e-6
    assert functional.arguments[-4].name == "q_head_num"
    assert functional.arguments[-4].default_value == 64
    assert str(functional.arguments[-4].type) == "int"
    assert functional.arguments[-3].name == "update_kv_cache"
    assert functional.arguments[-3].default_value is True
    assert str(functional.arguments[-3].type) == "bool"
    out_schema = torch._C.parse_schema(
        declarations[1].name + declarations[1].schema)
    assert out_schema.arguments[-5].name == "rms_norm_eps"
    assert out_schema.arguments[-4].name == "q_head_num"
    assert out_schema.arguments[-4].default_value == 64
    assert out_schema.arguments[-3].name == "update_kv_cache"
    assert out_schema.arguments[-3].default_value is True


def test_invalid_metadata_is_rejected_before_dispatch():
    for change in (
        {"x": meta((0, HIDDEN), torch.bfloat16)},
        {"w_qb": meta((Q_LORA, q_out_dim()), torch.int8)},
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
        w_qb_scale=torch.ones(q_out_dim(), dtype=torch.bfloat16),
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


def _seed_inputs(head_num):
    torch.manual_seed(0)
    x = (torch.randn(4, HIDDEN) * 0.05).to(torch.bfloat16)
    w_qa = (torch.randn(Q_LORA, HIDDEN) * 0.02).to(torch.bfloat16)
    gamma_qa = (torch.rand(Q_LORA) * 0.5 + 0.75).to(torch.bfloat16)
    w_qb = torch.randint(-4, 5, (Q_LORA, q_out_dim(head_num)), dtype=torch.int8)
    w_qb_scale = (torch.rand(q_out_dim(head_num)) * 0.004 + 0.002).to(
        torch.bfloat16)
    w_kv = (torch.randn(KV_LORA, HIDDEN) * 0.02).to(torch.bfloat16)
    gamma_kv = (torch.rand(KV_LORA) * 0.5 + 0.75).to(torch.bfloat16)
    angles = torch.rand(4, 1, 1, ROPE_DIM // 2) * 0.5
    rope_cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    rope_sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin


def test_qr_golden_is_the_bf16_rms_norm_row():
    x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin = (
        _seed_inputs(DEFAULT_HEAD_NUM))

    q_output, qr, kv_updates = cases.golden(
        torch, x, w_qa, gamma_qa, cases.pack_weight_nz(torch, w_qb),
        w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin)
    assert qr.shape == (4, Q_LORA)
    assert q_output.shape == (4, DEFAULT_HEAD_NUM, HEAD_DIM)
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


def test_cpu_golden_q_head_num_shapes_and_qr_independence():
    base = _seed_inputs(DEFAULT_HEAD_NUM)
    x, w_qa, gamma_qa, w_qb64, w_qb_scale64, w_kv, gamma_kv, rope_cos, rope_sin = base
    q_output64, qr64, _ = cases.golden(
        torch, x, w_qa, gamma_qa, cases.pack_weight_nz(torch, w_qb64),
        w_qb_scale64, w_kv, gamma_kv, rope_cos, rope_sin)
    for head_num in HEAD_NUM_CASES:
        q_out = q_out_dim(head_num)
        # NZ first dimension indexes 32-channel blocks, so the first
        # head_num*16 blocks are exactly the first head_num*512 channels.
        w_qb = cases.pack_weight_nz(torch, w_qb64[:, :q_out])
        q_output, qr, kv_updates = cases.golden(
            torch, x, w_qa, gamma_qa, w_qb, w_qb_scale64[:q_out],
            w_kv, gamma_kv, rope_cos, rope_sin, q_head_num=head_num)
        assert q_output.shape == (4, head_num, HEAD_DIM)
        assert kv_updates.shape == (4, KV_LORA)
        # qr never depends on the local head count (TP does not shard Q-LoRA).
        assert torch.equal(qr, qr64)
        # The Q-B channels are column-independent: any local head set must
        # agree exactly with the same channels of the H=64 oracle.
        torch.testing.assert_close(
            q_output, q_output64[:, :head_num, :], atol=0.0, rtol=0.0)


def test_case_parameters_cover_q_head_num_boundaries_and_both_modes():
    heads = {case.get("q_head_num", DEFAULT_HEAD_NUM)
             for case in cases.CASE_PARAMETERS}
    assert set(HEAD_NUM_CASES) <= heads
    assert len(cases.TOLERANCES) == len(OPERATOR.output_names) == 3
    modes = {case.get("update_kv_cache", True) for case in cases.CASE_PARAMETERS}
    assert modes == {True, False}
    # Both modes are exercised at both T boundaries for every head count.
    covered = {
        (case.get("q_head_num", DEFAULT_HEAD_NUM),
         case["t"], case.get("update_kv_cache", True))
        for case in cases.CASE_PARAMETERS
    }
    for head_num in HEAD_NUM_CASES:
        for t in (1, 96):
            for update in (True, False):
                assert (head_num, t, update) in covered, (
                    head_num, t, update)


def test_case_tolerances_match_outputs():
    assert len(cases.CASE_PARAMETERS) >= 1
    assert len(cases.TOLERANCES) == len(OPERATOR.output_names) == 3
    assert any("rms_norm_eps" in case for case in cases.CASE_PARAMETERS)
    for case in cases.CASE_PARAMETERS:
        assert cases.output_count(case) == (
            2 if case.get("update_kv_cache", True) else 3)
