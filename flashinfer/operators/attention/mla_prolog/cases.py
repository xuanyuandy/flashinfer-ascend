"""Independent CPU oracle and deterministic A3 acceptance inputs.

This file mirrors the frozen BF16 materialization points of the standalone
reference used during evolution: every public BF16 ingress, each GEMM output,
RMSNorm output, dequantized INT8 GEMM output and each RoPE pair is rounded to
BF16. The cache update is an exact scatter of the computed 512-value rows.
``qr`` is the same BF16 RMSNorm row that feeds dynamic quantization, and
``rms_norm_eps`` parameterizes both RMSNorm branches (default ``1e-6``).

The public KV cache is ``[22594, 128, 512]`` BF16 (about 2.76 GiB). Samples
allocate it directly on the NPU; the CPU oracle never materializes the dense
cache, it returns only the updated rows. Suites that cannot afford that device
allocation must add a maintainer-approved reduced-cache test path rather than
silently shrinking the public shape.
"""

from .operator import (
    CACHE_PAGES, CACHE_ROWS, HEADS, HEAD_DIM, HIDDEN, KV_LORA, MAX_T, MIN_T,
    NOPE_DIM, Q_LORA, Q_OUT, ROPE_DIM, W_QB_NZ_SHAPE,
)

SENTINEL_PAGE = CACHE_PAGES - 1
SENTINEL_ROW = CACHE_ROWS - 1


def bf16_round(torch, value):
    """Round FP32 values to BF16 with round-to-nearest-even, returned as FP32."""
    return value.to(torch.bfloat16).to(torch.float32)


def pack_weight_nz(torch, logical_weight):
    """Pack logical INT8 [K, N] into FRACTAL_NZ [N/32, K/16, 16, 32]."""
    logical_weight = logical_weight.to(torch.int8)
    if logical_weight.ndim != 2:
        raise ValueError("logical_weight must be rank-2 INT8")
    k, n = logical_weight.shape
    if k % 16 or n % 32:
        raise ValueError("logical_weight requires K % 16 == 0 and N % 32 == 0")
    return logical_weight.reshape(k // 16, 16, n // 32, 32).permute(2, 0, 1, 3).contiguous()


def unpack_weight_nz(torch, weight_nz, k, n):
    """Unpack FRACTAL_NZ [N/32, K/16, 16, 32] into logical INT8 [K, N]."""
    weight_nz = weight_nz.to(torch.int8)
    expected = (n // 32, k // 16, 16, 32)
    if tuple(weight_nz.shape) != expected:
        raise ValueError(f"weight_nz must be INT8 with shape {expected}")
    return weight_nz.permute(1, 2, 0, 3).reshape(k, n).contiguous()


def _cpu_float(tensor):
    return tensor.detach().to("cpu").float()


def _matmul_bf16(torch, left, right_rows):
    """FP32-accumulating [M,K] x [N,K]^T followed by BF16 materialization."""
    if not bool(left.any()) or not bool(right_rows.any()):
        return torch.zeros((left.shape[0], right_rows.shape[0]), dtype=torch.float32)
    result = torch.matmul(left, right_rows.T)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("matmul result contains non-finite values")
    return bf16_round(torch, result)


def _rms_norm_bf16(torch, value, gamma, epsilon=1.0e-6):
    value = value.to(torch.float32)
    gamma = gamma.to(torch.float32)
    mean_square = (value * value).sum(dim=-1, dtype=torch.float32) / value.shape[-1]
    reciprocal_rms = 1.0 / torch.sqrt(mean_square + float(epsilon))
    normalized = value * reciprocal_rms.unsqueeze(-1) * gamma
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("rms_norm result contains non-finite values")
    return bf16_round(torch, normalized)


def _dynamic_quant(torch, value):
    value = value.to(torch.float32)
    row_max = value.abs().amax(dim=-1)
    scale = row_max / 127.0
    quantized = torch.zeros(value.shape, dtype=torch.int8)
    nonzero = scale != 0
    if bool(nonzero.any()):
        rounded = torch.round(value[nonzero] / scale[nonzero].unsqueeze(-1))
        quantized[nonzero] = rounded.clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _quant_matmul_bf16(torch, value, weight, per_token_scale, weight_scale, column_chunk):
    rows = value.shape[0]
    result = torch.empty((rows, weight.shape[1]), dtype=torch.float32)
    if not bool(value.any()) or not bool(weight.any()):
        result.fill_(0.0)
        return result
    value_f64 = value.to(torch.float64)
    for begin in range(0, weight.shape[1], column_chunk):
        end = min(begin + column_chunk, weight.shape[1])
        accumulator = torch.matmul(value_f64, weight[:, begin:end].to(torch.float64))
        dequant = accumulator.to(torch.float32)
        dequant = dequant * per_token_scale.unsqueeze(-1)
        dequant = dequant * weight_scale[begin:end].unsqueeze(0)
        if not bool(torch.isfinite(dequant).all()):
            raise ValueError("quant matmul result contains non-finite values")
        result[:, begin:end] = bf16_round(torch, dequant)
    return result


def _partial_rope_bf16(torch, value, rope_cos, rope_sin):
    """Rotate interleaved even/odd pairs in the last D_ROPE head channels."""
    value = value.to(torch.float32)
    t = value.shape[0]
    result = value.clone()
    tail = value[..., NOPE_DIM:NOPE_DIM + ROPE_DIM]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    cos = rope_cos.to(torch.float32).reshape(t, ROPE_DIM)[:, 0::2]
    sin = rope_sin.to(torch.float32).reshape(t, ROPE_DIM)[:, 0::2]
    broadcast = (t,) + (1,) * (value.dim() - 2) + (ROPE_DIM // 2,)
    cos, sin = cos.reshape(broadcast), sin.reshape(broadcast)
    rotated = torch.empty_like(tail)
    rotated[..., 0::2] = bf16_round(torch, even * cos - odd * sin)
    rotated[..., 1::2] = bf16_round(torch, even * sin + odd * cos)
    result[..., NOPE_DIM:NOPE_DIM + ROPE_DIM] = rotated
    if not bool(torch.isfinite(result).all()):
        raise ValueError("rope result contains non-finite values")
    return result


def golden_kv_updates(torch, x, w_kv, gamma_kv, rope_cos, rope_sin, *,
                      row_chunk=8, rms_norm_eps=1.0e-6):
    """Compute only the 512-value cache rows materialized by the call."""
    x = bf16_round(torch, _cpu_float(x))
    w_kv = bf16_round(torch, _cpu_float(w_kv))
    gamma_kv = bf16_round(torch, _cpu_float(gamma_kv))
    rope_cos, rope_sin = _cpu_float(rope_cos), _cpu_float(rope_sin)
    t = x.shape[0]
    updates = torch.empty((t, KV_LORA), dtype=torch.float32)
    for begin in range(0, t, row_chunk):
        end = min(begin + row_chunk, t)
        kv = _matmul_bf16(torch, x[begin:end], w_kv)
        kv_norm = _rms_norm_bf16(torch, kv, gamma_kv, rms_norm_eps)
        updates[begin:end] = _partial_rope_bf16(
            torch, kv_norm, rope_cos[begin:end], rope_sin[begin:end])
    return updates


def golden(torch, x, w_qa, gamma_qa, w_qb_nz, w_qb_scale, w_kv, gamma_kv,
           rope_cos, rope_sin, *, row_chunk=8, column_chunk=2048,
           rms_norm_eps=1.0e-6):
    """Return ``(q_output, qr, kv_updates)`` for the frozen nine-op chain.

    ``kv_cache`` is not passed: the oracle returns the rows the kernel scatters
    and the caller applies them to the cache representation it owns.
    """
    x = bf16_round(torch, _cpu_float(x))
    w_qa = bf16_round(torch, _cpu_float(w_qa))
    gamma_qa = bf16_round(torch, _cpu_float(gamma_qa))
    w_qb_scale = bf16_round(torch, _cpu_float(w_qb_scale))
    w_kv = bf16_round(torch, _cpu_float(w_kv))
    gamma_kv = bf16_round(torch, _cpu_float(gamma_kv))
    rope_cos, rope_sin = _cpu_float(rope_cos), _cpu_float(rope_sin)
    w_qb = unpack_weight_nz(torch, w_qb_nz.detach().to("cpu"), Q_LORA, Q_OUT)
    t = x.shape[0]
    if t < MIN_T or t > MAX_T:
        raise ValueError("T must be in [1, 4096]")

    q_output = torch.empty((t, HEADS, HEAD_DIM), dtype=torch.float32)
    qr = torch.empty((t, Q_LORA), dtype=torch.float32)
    kv_updates = torch.empty((t, KV_LORA), dtype=torch.float32)
    for begin in range(0, t, row_chunk):
        end = min(begin + row_chunk, t)
        x_chunk = x[begin:end]
        qa = _matmul_bf16(torch, x_chunk, w_qa)
        qa_norm = _rms_norm_bf16(torch, qa, gamma_qa, rms_norm_eps)
        qr[begin:end] = qa_norm
        qa_int8, qa_scale = _dynamic_quant(torch, qa_norm)
        q_raw = _quant_matmul_bf16(
            torch, qa_int8, w_qb, qa_scale, w_qb_scale, column_chunk,
        ).reshape(end - begin, HEADS, HEAD_DIM)
        q_output[begin:end] = _partial_rope_bf16(
            torch, q_raw, rope_cos[begin:end], rope_sin[begin:end])

        kv = _matmul_bf16(torch, x_chunk, w_kv)
        kv_norm = _rms_norm_bf16(torch, kv, gamma_kv, rms_norm_eps)
        kv_updates[begin:end] = _partial_rope_bf16(
            torch, kv_norm, rope_cos[begin:end], rope_sin[begin:end])
    return q_output, qr, kv_updates


def sentinel_row(torch):
    values = torch.arange(KV_LORA, dtype=torch.float32) / 512.0
    return bf16_round(torch, values)


def sample(torch, torch_npu, *, t=1, rms_norm_eps=1.0e-6):
    """Deterministic A3 input. The dense cache is allocated on the NPU."""
    rng = torch.Generator().manual_seed(20261117 + int(t))
    dtype = torch.bfloat16

    def normal(shape, scale):
        return (torch.randn(shape, generator=rng) * scale).to(dtype)

    x = normal((t, HIDDEN), 0.05)
    w_qa = normal((Q_LORA, HIDDEN), 0.02)
    gamma_qa = (torch.rand((Q_LORA,), generator=rng) * 0.5 + 0.75).to(dtype)
    w_qb_logical = torch.randint(-4, 5, (Q_LORA, Q_OUT), generator=rng, dtype=torch.int8)
    w_qb = pack_weight_nz(torch, w_qb_logical)
    w_qb_scale = (torch.rand((Q_OUT,), generator=rng) * 0.004 + 0.002).to(dtype)
    w_kv = normal((KV_LORA, HIDDEN), 0.02)
    gamma_kv = (torch.rand((KV_LORA,), generator=rng) * 0.5 + 0.75).to(dtype)
    angles = torch.rand((t, 1, 1, ROPE_DIM // 2), generator=rng) * 0.5
    rope_cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    rope_sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    slots = torch.stack(
        (torch.arange(t) // CACHE_ROWS, torch.arange(t) % CACHE_ROWS), dim=1,
    ).to(torch.int32)

    def nd(tensor):
        # TorchNPU labels four-dimensional tensors NCHW by default.
        return torch_npu.npu_format_cast(tensor.npu(), 2) if tensor.dim() == 4 else tensor.npu()

    x_device = nd(x)
    cache = torch.zeros(
        (CACHE_PAGES, CACHE_ROWS, KV_LORA), dtype=dtype, device=x_device.device)
    cache[SENTINEL_PAGE, SENTINEL_ROW].copy_(sentinel_row(torch).to(cache.device))
    args = (
        x_device, nd(w_qa), nd(gamma_qa),
        torch_npu.npu_format_cast(w_qb.npu(), 29), nd(w_qb_scale),
        nd(w_kv), nd(gamma_kv), nd(rope_cos), nd(rope_sin), nd(slots), cache,
    )
    q_output, qr, _ = golden(
        torch, x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, rms_norm_eps=rms_norm_eps,
    )
    expected = (q_output, qr)
    options = {} if float(rms_norm_eps) == 1.0e-6 else dict(
        rms_norm_eps=float(rms_norm_eps))
    return args, options, expected


CASE_PARAMETERS = (
    dict(t=1),
    dict(t=96),
    dict(t=193),
    # Exercise the keyword attribute end to end with the DS4.1 value.
    dict(t=96, rms_norm_eps=1.0e-20),
)
# BF16 policy v2: matched ratio >= 0.999, atol/rtol = 0.0156, max_abs <= 1.0.
# torch.testing.assert_close consumes the (atol, rtol) pair per output; the
# tuple follows OUTPUT_NAMES (q_output, qr).
TOLERANCES = ((0.0156, 0.0156), (0.0156, 0.0156))


def verify_effect(torch, call, args, options, outputs):
    """Check the in-place cache effect the fresh outputs cannot express."""
    (cache,) = (args[-1],)
    slots = args[-2].detach().to(cache.device).long()
    sentinel = cache[SENTINEL_PAGE, SENTINEL_ROW].detach().to("cpu", torch.float32)
    if not torch.equal(sentinel, sentinel_row(torch)):
        raise AssertionError("kv_cache sentinel row was modified")
    updates = golden_kv_updates(
        torch, args[0], args[5], args[6], args[7], args[8],
        rms_norm_eps=float(options.get("rms_norm_eps", 1.0e-6)))
    actual = cache[slots[:, 0], slots[:, 1]].detach().to("cpu", torch.float32)
    torch.testing.assert_close(actual, updates, atol=0.0156, rtol=0.0156)
