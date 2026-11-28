"""Independent CPU oracle and samples for the dense IndexerPrologueQW contract.

This module deliberately imports no torch_npu and performs no device execution.
Weights are synthesized deterministically; the oracle expresses the production
W8A8 path in CPU Torch (per-row DynamicQuant, integer accumulation, dequant,
adjacent-pair RoPE and the FP16 scale round trip).
"""

import math

from .operator import (
    DEFAULT_SOFTMAX_SCALE,
    DIM_Q,
    DIM_QR,
    DIM_X,
    HEAD_DIM,
    N_HEADS,
    ROPE_DIM,
)


def _dynamic_quant(torch, matrix):
    """Per-row BF16 -> FP32 DynamicQuant with the kernel's published scale."""
    values = matrix.float()
    amax = values.abs().amax(dim=1)
    positive = amax > 0
    safe = torch.where(positive, amax, torch.ones_like(amax))
    multiplier = 127.0 / safe
    quantized = torch.round(values * multiplier[:, None]).clamp(-127, 127).to(torch.int8)
    scale = torch.where(positive, 1.0 / multiplier, torch.zeros_like(amax)).float()
    return quantized, scale


def _rope_tail(torch, q, rope_sin, rope_cos):
    tail = q[..., -ROPE_DIM:].float()
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rotated = torch.empty_like(tail)
    rotated[..., 0::2] = even * rope_cos[:, None, 0::2] - odd * rope_sin[:, None, 0::2]
    rotated[..., 1::2] = odd * rope_cos[:, None, 1::2] + even * rope_sin[:, None, 1::2]
    return torch.cat([q[..., :-ROPE_DIM], rotated.to(torch.bfloat16)], dim=-1)


def _quantize_query(torch, q):
    amax = q.float().abs().amax(dim=-1, keepdim=True)
    scale16 = (amax / 127.0).clamp_min(2 ** -24).to(torch.float16)
    divisor = scale16.float()
    q8 = torch.round(q.float() / divisor).clamp(-127, 127).to(torch.int8)
    return q8, scale16.squeeze(-1)


def _w_projection(torch, x, ww, softmax_scale):
    projected = torch.matmul(x.to(torch.bfloat16), ww.t().to(torch.bfloat16))
    return (projected.float() * (softmax_scale * N_HEADS ** -0.5)).to(torch.float16)


def golden(torch, x, qr, wqb, ww, rope_sin, rope_cos,
           softmax_scale=DEFAULT_SOFTMAX_SCALE):
    """CPU-only production-order reference for the dense public entry."""
    tokens = qr.shape[0]
    weight_q, weight_scale = _dynamic_quant(torch, wqb)
    activation_q, activation_scale = _dynamic_quant(torch, qr)
    accumulated = torch.matmul(activation_q.int(), weight_q.int().t())
    dequant = (accumulated.float() * (activation_scale[:, None] * weight_scale[None, :]))
    projected = dequant.to(torch.bfloat16).view(tokens, N_HEADS, HEAD_DIM)
    roped = _rope_tail(torch, projected, rope_sin, rope_cos)
    q8, q_scale = _quantize_query(torch, roped)
    w = _w_projection(torch, x, ww, softmax_scale)
    return q8, q_scale, w


def _rope_tables(torch, tokens, generator):
    positions = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    frequencies = 1.0 / (
        40000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    base = (positions * frequencies.unsqueeze(0)).repeat_interleave(2, dim=-1)
    angles = base + torch.rand((tokens, ROPE_DIM), generator=generator) * 0.25
    return torch.sin(angles), torch.cos(angles)


def _unit_amax(torch, tensor):
    """Normalize each BF16 row to amax=1 so DynamicQuant has no half-way ties.

    The production vector quantizer may differ by one ULP from IEEE division at
    exact .5 boundaries. This independent CPU oracle stays exact (and keeps the
    strict INT8 tolerance) by exercising non-degenerate, contract-valid rows.
    """
    values = tensor.float()
    amax = values.abs().amax(dim=1, keepdim=True)
    return (values / amax).to(torch.bfloat16)


def prequantized_golden(torch, x, qr, weight_q, weight_scale, ww, rope_sin, rope_cos,
                        softmax_scale=DEFAULT_SOFTMAX_SCALE):
    """CPU reference over the same logical INT8 ND weight the launch consumes."""
    tokens = qr.shape[0]
    activation_q, activation_scale = _dynamic_quant(torch, qr)
    accumulated = torch.matmul(activation_q.int(), weight_q.int())
    dequant = (accumulated.float() * (activation_scale[:, None] * weight_scale[None, :]))
    projected = dequant.to(torch.bfloat16).view(tokens, N_HEADS, HEAD_DIM)
    roped = _rope_tail(torch, projected, rope_sin, rope_cos)
    q8, q_scale = _quantize_query(torch, roped)
    w = _w_projection(torch, x, ww, softmax_scale)
    return q8, q_scale, w


def sample(torch, torch_npu, *, tokens=48, seed=20261119,
           softmax_scale=DEFAULT_SOFTMAX_SCALE, prequantized=False):
    generator = torch.Generator(device="cpu").manual_seed(seed + tokens)
    x = torch.randn(tokens, DIM_X, generator=generator).to(torch.bfloat16)
    qr = _unit_amax(torch, torch.randn(tokens, DIM_QR, generator=generator).to(torch.bfloat16))
    wqb = _unit_amax(torch, (torch.randn(DIM_Q, DIM_QR, generator=generator)
                             / math.sqrt(DIM_QR)).to(torch.bfloat16))
    ww = (torch.randn(N_HEADS, DIM_X, generator=generator) / math.sqrt(DIM_X)).to(torch.bfloat16)
    rope_sin, rope_cos = _rope_tables(torch, tokens, generator)
    if prequantized:
        from flashinfer.attention.indexer_prologue import prepare_indexer_prologue_qw_weight

        wqb_nz, wqb_scale = prepare_indexer_prologue_qw_weight(wqb.npu())
        weight_q, weight_scale = _dynamic_quant(torch, wqb)
        logical_weight = weight_q.t().contiguous()
        expected = prequantized_golden(
            torch, x, qr, logical_weight, weight_scale, ww, rope_sin, rope_cos,
            softmax_scale)
        args = (x.npu(), qr.npu(), wqb_nz, ww.npu(), rope_sin.npu(), rope_cos.npu())
        return args, dict(softmax_scale=softmax_scale, wqb_scale=wqb_scale,
                          prequantized=True), expected
    expected = golden(torch, x, qr, wqb, ww, rope_sin, rope_cos, softmax_scale)
    args = (x.npu(), qr.npu(), wqb.npu(), ww.npu(), rope_sin.npu(), rope_cos.npu())
    return args, dict(softmax_scale=softmax_scale), expected


DENSE_CASE_PARAMETERS = tuple(
    dict(tokens=tokens, seed=20261119)
    for tokens in (1, 3, 48, 49, 72, 96, 128, 129)
)
PREQUANTIZED_CASE_PARAMETERS = tuple(
    dict(tokens=tokens, seed=20261119, prequantized=True)
    for tokens in (1, 3, 48, 49, 72, 96, 128, 129)
)
CASE_PARAMETERS = DENSE_CASE_PARAMETERS + PREQUANTIZED_CASE_PARAMETERS


def parameters_for(kernel_id):
    return (PREQUANTIZED_CASE_PARAMETERS if "prequantized" in kernel_id
            else DENSE_CASE_PARAMETERS)


TOLERANCES = ((1.0, 0.0), (9e-2, 2 ** -10), (9e-2, 2 ** -10))
