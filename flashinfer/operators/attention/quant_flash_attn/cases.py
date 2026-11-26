"""Independent constant-score/constant-V golden for the C4A4 native package."""

import math

from .operator import kv_shapes


def sample(torch, torch_npu, *, dim=128, kv_layout="PA_BBND", qscale_layout="TND",
           causal=False, schema=0, lse=True, kv_length=63, heads=4):
    device = torch.device("npu", torch.npu.current_device())
    page = 128
    pages = (kv_length + page - 1) // page
    tokens, kv_heads = 3, 2

    def packed(shape, byte, dtype):
        storage = torch.full((math.prod(shape),), byte, dtype=torch.uint8, device=device)
        return storage.view(dtype).reshape(shape)

    kv_shape, ks_shape, vs_shape = kv_shapes(kv_layout, pages, page, kv_heads, dim)
    qs_shape = (tokens, heads, dim // 64, 2) if qscale_layout == "TND" else (
        kv_heads, tokens, heads // kv_heads, dim // 64, 2)
    fp4, e8m0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    q = packed((tokens, heads, dim // 2), 0, fp4)
    k, v = packed(kv_shape, 0, fp4), packed(kv_shape, 0x22, fp4)
    qs, ks, vs = (packed(shape, 127, e8m0) for shape in (qs_shape, ks_shape, vs_shape))
    cu_q = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    used_kv = torch.tensor([kv_length], dtype=torch.int32, device=device)
    mask_mode = 3 if causal else 0
    layout = dict(layout_kv=kv_layout, layout_q_descale=qscale_layout, mask_mode=mask_mode)
    # The query fits one M tile per head (prefill) or KV head (decode).
    # Give each of those tiles one full-KV FA task so the main package can be
    # verified with an externally supplied schedule, including for long KV.
    props = torch.npu.get_device_properties(device.index)
    metadata = torch.zeros((2, 4096), dtype=torch.int32)
    metadata[0, :6] = torch.tensor([1, 0, 128, 256 if dim == 128 else 128,
                                    props.cube_core_num, props.vector_core_num], dtype=torch.int32)
    groups = heads if qscale_layout == "TND" else kv_heads
    for index in range(groups):
        record = 16 * (1 + index)
        metadata[0, record] = index
        metadata[0, record + 3] = index + 1
    options = dict(block_table=torch.arange(pages, dtype=torch.int32, device=device).reshape(1, -1),
                   cu_seqlens_q=cu_q, seqused_kv=used_kv, metadata=metadata.to(device), softmax_scale=dim**-0.5,
                   return_softmax_lse=lse, quant_schema=schema, **layout)
    if causal:
        options["attn_mask"] = torch.ones((2048, 2048), dtype=torch.int8, device=device).triu(1)
    # FP4 E2M1 code 2 with E8M0 byte 127 represents one. Q/K are all zero,
    # so attention is the average of ones, independent of P quantization.
    visible = torch.arange(kv_length - tokens + 1, kv_length + 1) if causal else torch.full((tokens,), kv_length)
    expected = torch.ones((tokens, heads, dim), dtype=torch.bfloat16)
    # OCP mode represents the zero-score exponent exactly.
    expected_lse = visible.float().log().expand(heads, tokens).contiguous() if lse else torch.empty(0)
    return (q, k, v, qs, ks, vs), options, (expected, expected_lse)


CASE_PARAMETERS = (
    dict(dim=128, kv_layout="PA_BBND", qscale_layout="TND"),
    dict(dim=256, kv_layout="PA_BNBD", qscale_layout="N2TGD", causal=True),
    dict(dim=128, kv_layout="PA_NZ", qscale_layout="N2TGD", causal=True),
    dict(dim=256, kv_layout="PA_NZ", qscale_layout="TND", kv_length=4096),
    dict(dim=256, kv_layout="PA_BNBD", schema=1, lse=False),
    dict(dim=128, kv_layout="PA_BBND", schema=2, lse=False),
)
TOLERANCES = ((0.005, 0.005), (0.001, 0.001))
