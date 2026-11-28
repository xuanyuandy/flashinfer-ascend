"""Independent CPU oracle and generic cases for mutation-only K prologue.

The oracle imports no torch_npu and performs no device computation. The full
cache backing is the comparison unit: slots outside valid rows, padding entries
and storage gaps must retain their original values.
"""

import math

HEAD_DIM = 128
ROPE_DIM = 64
INV_HEAD_DIM = 1.0 / HEAD_DIM
SCALE_FLOOR = 1e-12


def _full_storage_view(torch, tensor):
    elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    return torch.as_strided(tensor, (elements,), (1,), 0)


def golden(torch, latent, wk, norm_weight, rope_cos, rope_sin, indices,
           k_cache, k_scale_cache, norm_eps):
    """Return updated full cache backings on CPU, including untouched gaps."""
    tokens = int(latent.shape[0])
    blocks, slots = int(k_cache.shape[0]), int(k_cache.shape[1])
    key_storage = _full_storage_view(torch, k_cache).clone()
    scale_storage = _full_storage_view(torch, k_scale_cache).clone()
    updated_key = torch.as_strided(key_storage, k_cache.shape, k_cache.stride(),
                                   k_cache.storage_offset())
    updated_scale = torch.as_strided(scale_storage, k_scale_cache.shape,
                                     k_scale_cache.stride(), k_scale_cache.storage_offset())
    if tokens == 0:
        return key_storage, scale_storage

    projected = torch.matmul(latent.float(), wk.float().t()).to(torch.bfloat16)
    gamma = norm_weight.float()
    pairs = indices.detach().cpu().tolist()
    for row in range(tokens):
        normalized_bf16 = projected[row]
        values = normalized_bf16.float()
        variance = (values * values).mean()
        rstd = 1.0 / math.sqrt(float(variance) + float(norm_eps))
        normalized = (values * rstd * gamma).to(torch.bfloat16)

        # The kernel reduces amax and divides the first 64 lanes from the
        # normalized BF16 row; the tail is the raw FP32 RoPE result.
        head = normalized[:ROPE_DIM].float()
        tail = normalized[-ROPE_DIM:].float()
        even, odd = tail[0::2], tail[1::2]
        sin = rope_sin[row].reshape(ROPE_DIM)
        cos = rope_cos[row].reshape(ROPE_DIM)
        rotated = torch.empty(ROPE_DIM, dtype=torch.float32)
        rotated[0::2] = even * cos[0::2] - odd * sin[0::2]
        rotated[1::2] = odd * cos[1::2] + even * sin[1::2]
        # amax uses the BF16 RoPE round trip; division uses raw FP32 RoPE.
        amax_head = float(head.abs().max())
        amax_tail = float(rotated.to(torch.bfloat16).float().abs().max())
        scale = max(amax_head, amax_tail) / 127.0
        scale = max(scale, SCALE_FLOOR)

        quantized = torch.empty(HEAD_DIM, dtype=torch.int8)
        quantized[:ROPE_DIM] = torch.trunc(head / scale).clamp(-127, 127).to(torch.int8)
        quantized[ROPE_DIM:] = torch.trunc(rotated / scale).clamp(-127, 127).to(torch.int8)

        block, slot = int(pairs[row][0]), int(pairs[row][1])
        if block == -1 and slot == -1:
            continue
        if block < 0 or slot < 0 or block >= blocks or slot >= slots:
            raise ValueError("golden received an unsupported cache address")
        updated_key[block, slot, 0, :] = quantized
        updated_scale[block, slot, 0, 0] = torch.tensor(scale, dtype=torch.float32).to(torch.float16)
    return key_storage, scale_storage


def make_inputs(torch, *, tokens=17, blocks=5, slots=7, key_stride=None,
                scale_stride=None, norm_dtype=None, seed=20261119):
    """Deterministic CPU inputs plus deliberately strided cache backings."""
    norm_dtype = torch.bfloat16 if norm_dtype is None else norm_dtype
    generator = torch.Generator(device="cpu").manual_seed(seed + tokens)
    latent = torch.randn(tokens, 512, generator=generator).to(torch.bfloat16)
    wk = (torch.randn(HEAD_DIM, 512, generator=generator) / math.sqrt(512)).to(torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, generator=generator)
    norm_weight = (norm_weight if norm_dtype == torch.float32
                   else norm_weight.to(torch.bfloat16))
    rope_sin = torch.randn(tokens, 1, 1, ROPE_DIM, generator=generator)
    rope_cos = torch.randn(tokens, 1, 1, ROPE_DIM, generator=generator)
    # Tests stay inside the supported input domain: valid rows use distinct
    # (block, slot) addresses. Duplicate slots are not checked or promised.
    pad = tokens >= 3
    valid = tokens - (2 if pad else 0)
    blocks = max(blocks, (valid + slots - 1) // slots)
    addresses = [(block, slot) for block in range(blocks) for slot in range(slots)]
    chosen = torch.randperm(len(addresses), generator=generator)[:valid].tolist()
    rows = [addresses[index] for index in chosen]
    indices = torch.zeros(tokens, 2, dtype=torch.int32)
    for offset, (block, slot) in enumerate(rows):
        indices[offset + (1 if pad else 0)] = torch.tensor([block, slot], dtype=torch.int32)
    # Exact [-1,-1] padding rows remain in the supported domain.
    if pad:
        indices[0] = torch.tensor([-1, -1], dtype=torch.int32)
        indices[-1] = torch.tensor([-1, -1], dtype=torch.int32)
    key_block_stride = slots * HEAD_DIM + 9 if key_stride is None else key_stride
    scale_block_stride = slots + 3 if scale_stride is None else scale_stride
    key_storage = torch.randint(
        -128, 127, (max(blocks * key_block_stride, HEAD_DIM) + 11,),
        generator=generator, dtype=torch.int8)
    scale_storage = torch.randn(max(blocks * scale_block_stride, 1) + 5,
                                generator=generator).to(torch.float16)
    key_view = torch.as_strided(key_storage, (blocks, slots, 1, HEAD_DIM),
                                (key_block_stride, HEAD_DIM, HEAD_DIM, 1), 3)
    scale_view = torch.as_strided(scale_storage, (blocks, slots, 1, 1),
                                  (scale_block_stride, 1, 1, 1), 2)
    expected = golden(torch, latent, wk, norm_weight, rope_cos, rope_sin, indices,
                      key_view, scale_view, 1e-6)
    return dict(latent=latent, wk=wk, norm_weight=norm_weight, rope_sin=rope_sin,
                rope_cos=rope_cos, indices=indices, k_cache=key_view,
                k_scale_cache=scale_view, norm_eps=1e-6), expected


def sample(torch, torch_npu, *, tokens=17, blocks=5, slots=7, key_stride=None,
           scale_stride=None, norm_dtype="bfloat16", seed=20261119):
    """Move one deterministic CPU case to NPU without copying cache views."""
    if isinstance(norm_dtype, str):
        norm_dtype = getattr(torch, norm_dtype)
    cpu, expected = make_inputs(
        torch, tokens=tokens, blocks=blocks, slots=slots, key_stride=key_stride,
        scale_stride=scale_stride, norm_dtype=norm_dtype, seed=seed)
    key_cpu, scale_cpu = cpu["k_cache"], cpu["k_scale_cache"]
    key_storage = _full_storage_view(torch, key_cpu).npu()
    scale_storage = _full_storage_view(torch, scale_cpu).npu()
    key = torch.as_strided(key_storage, key_cpu.shape, key_cpu.stride(),
                           key_cpu.storage_offset())
    scale = torch.as_strided(scale_storage, scale_cpu.shape, scale_cpu.stride(),
                             scale_cpu.storage_offset())
    args = (cpu["latent"].npu(), cpu["wk"].npu(), cpu["norm_weight"].npu(),
            cpu["rope_cos"].npu(), cpu["rope_sin"].npu(), cpu["indices"].npu(),
            key, scale)
    return args, dict(norm_eps=cpu["norm_eps"]), expected


def verify_mutation(torch, args, expected):
    """Verify complete backings; written keys allow 1 ULP, gaps are exact."""
    actual_key = _full_storage_view(torch, args[6]).cpu()
    actual_scale = _full_storage_view(torch, args[7]).cpu()
    expected_key, expected_scale = expected
    torch.testing.assert_close(actual_key.int(), expected_key.int(), atol=1, rtol=0)
    torch.testing.assert_close(actual_scale.double(), expected_scale.double(), atol=0, rtol=0)

    key = args[6]
    written = torch.zeros(expected_key.numel(), dtype=torch.bool)
    for block, slot in args[5].cpu().tolist():
        if block == slot == -1:
            continue
        base = key.storage_offset() + block * key.stride()[0] + slot * key.stride()[1]
        written[base + torch.arange(HEAD_DIM)] = True
    torch.testing.assert_close(
        actual_key.int()[~written], expected_key.int()[~written], atol=0, rtol=0)
    return (
        float((actual_key.int() - expected_key.int()).abs().max()),
        float((actual_scale.double() - expected_scale.double()).abs().max()),
    )


CASE_PARAMETERS = tuple(
    dict(tokens=tokens, norm_dtype=norm_dtype, seed=20261119)
    for tokens in (1, 4, 5, 16, 17, 48)
    for norm_dtype in ("bfloat16", "float32")
) + (
    dict(tokens=1, blocks=1, slots=1, key_stride=0, scale_stride=0,
         norm_dtype="bfloat16", seed=20261119),
)
TOLERANCES = ()
