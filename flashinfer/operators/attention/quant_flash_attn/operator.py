"""Public contract for Ascend950 MXFP4 GQA with FP32 softmax."""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec

OP = "attention.quant_flash_attn"
FP4 = "float4_e2m1fn_x2"
E8M0 = "float8_e8m0fnu"
KV_LAYOUTS = ("PA_BBND", "PA_BNBD", "PA_NZ")
INPUTS = (
    "q", "k", "v", "q_descale", "k_descale", "v_descale",
    "block_table", "p_scale", "cu_seqlens_q", "cu_seqlens_kv",
    "seqused_q", "seqused_kv", "sinks", "attn_mask", "metadata",
    "v_tail", "block_table_tail", "seqused_v_tail",
)
ATTRIBUTES = (
    "quant_compute_mode", "softmax_scale", "mask_mode", "win_left", "win_right",
    "max_seqlen_q", "max_seqlen_kv", "layout_q", "layout_q_descale", "layout_kv",
    "layout_out", "return_softmax_lse", "quant_schema",
)


def check_layouts(layout_q, layout_q_descale, layout_kv, layout_out, mask_mode):
    if layout_q != "TND" or layout_out != "TND":
        raise ValueError("C4A4 requires TND query and output")
    if layout_q_descale not in ("TND", "N2TGD") or layout_kv not in KV_LAYOUTS:
        raise ValueError("C4A4 requires TND/N2TGD Q descale and PA_BBND/PA_BNBD/PA_NZ KV")
    if mask_mode not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3 for C4A4")


def kv_shapes(layout, pages, page, heads, dim):
    if layout == "PA_BBND":
        return (pages, page, heads, dim // 2), (pages, page, heads, dim // 64, 2), (
            pages, page // 64, heads, dim, 2)
    if layout == "PA_BNBD":
        return (pages, heads, page, dim // 2), (pages, heads, page, dim // 64, 2), (
            pages, heads, page // 64, dim, 2)
    return (pages, heads, dim // 64, page, 32), (pages, heads, page // 16, dim // 64, 16, 2), (
        pages, heads, dim // 16, page // 64, 16, 2)


def check_tensor(tensor, name, expected_dtype, expected_shape):
    if tuple(tensor.shape) != expected_shape or str(tensor.dtype).removeprefix("torch.") != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype} and shape {expected_shape}")


def infer(arguments, layouts):
    q, k, v, qs, ks, vs = (arguments[name] for name in INPUTS[:6])
    table, cu_q, used_q, used_kv, mask, metadata = (
        arguments[name] for name in ("block_table", "cu_seqlens_q", "seqused_q", "seqused_kv", "attn_mask", "metadata")
    )
    layout = arguments["layout_kv"]
    qscale_layout = arguments["layout_q_descale"]
    mask_mode = arguments["mask_mode"]
    check_layouts(arguments["layout_q"], qscale_layout, layout, arguments["layout_out"], mask_mode)
    if (arguments["quant_mode"] != 5 or arguments["quant_schema"] not in (0, 1, 2)
            or arguments["win_left"] != -1 or arguments["win_right"] != -1):
        raise ValueError("C4A4 requires quant_mode=5, quant_schema=0..2 and unrestricted windows")
    scale = arguments["softmax_scale"]
    if not math.isfinite(scale) or not 0 < scale <= 3.4028234663852886e38:
        raise ValueError("softmax_scale must be positive, finite and representable as float32")
    if any(n != -1 and not 0 < n <= 2**31 - 1 for n in (
            arguments["max_seqlen_q"], arguments["max_seqlen_kv"])):
        raise ValueError("maximum lengths must be positive or -1")

    if q.ndim != 3 or k.ndim != (5 if layout == "PA_NZ" else 4):
        raise ValueError("invalid query or paged KV rank")
    tokens, q_heads, packed_dim = q.shape
    dim = 2 * packed_dim
    pages = k.shape[0]
    if layout == "PA_BBND":
        page, kv_heads = k.shape[1:3]
    else:
        kv_heads, page = k.shape[1], k.shape[-2]
    if (dim not in (128, 256) or page not in (64, 128, 256, 512, 1024)
            or min(tokens, q_heads, pages, kv_heads) <= 0 or q_heads % kv_heads):
        raise ValueError("invalid D, page size or GQA head counts")
    groups = q_heads // kv_heads
    if q_heads > 256 or kv_heads > 256 or groups > 64:
        raise ValueError("C4A4 supports at most 256 Q and KV heads and 64 Q heads per KV head")
    kv_shape, ks_shape, vs_shape = kv_shapes(layout, pages, page, kv_heads, dim)
    qs_shape = (tokens, q_heads, dim // 64, 2) if qscale_layout == "TND" else (
        kv_heads, tokens, groups, dim // 64, 2)
    for tensor, name, dtype, shape in (
        (q, "q", FP4, (tokens, q_heads, packed_dim)), (k, "k", FP4, kv_shape), (v, "v", FP4, kv_shape),
        (qs, "q_descale", E8M0, qs_shape), (ks, "k_descale", E8M0, ks_shape), (vs, "v_descale", E8M0, vs_shape),
    ):
        check_tensor(tensor, name, dtype, shape)
    if table.ndim != 2 or min(table.shape) <= 0 or page * table.shape[1] > 2**31 - 1:
        raise ValueError("block_table requires positive [B, Pmax] shape")
    batch = table.shape[0]
    if batch >= 65536:
        raise ValueError("C4A4 batch size must be less than 65536")
    check_tensor(table, "block_table", "int32", tuple(table.shape))
    check_tensor(cu_q, "cu_seqlens_q", "int32", (batch + 1,))
    check_tensor(used_kv, "seqused_kv", "int32", (batch,))
    if used_q is not None:
        check_tensor(used_q, "seqused_q", "int32", (batch,))
    if metadata.ndim not in (1, 2) or (metadata.ndim == 2 and metadata.shape[0] != 2) or metadata.shape[-1] < 16:
        raise ValueError("metadata requires int32 [C] or [2,C] with a complete header")
    check_tensor(metadata, "metadata", "int32", tuple(metadata.shape))
    if mask_mode == 3 and mask is not None:
        check_tensor(mask, "attn_mask", "int8", (2048, 2048))
    elif mask_mode == 3 or mask is not None:
        raise ValueError("provide a causal mask template only with mask_mode=3")

    tensors = tuple(arguments.get(name) for name in INPUTS)
    if layouts is not None and (any(kind != "ND" for kind in layouts.values()) or any(
            not tensor.is_contiguous()
            for name, tensor in zip(INPUTS, tensors)
            if tensor is not None and name not in {"k", "v", "k_descale", "v_descale"})):
        raise ValueError("C4A4 requires physical ND and contiguous inputs except at KV page/head axes")
    attrs = {name: arguments[name] for name in ATTRIBUTES[1:]}
    attrs["quant_compute_mode"] = arguments["quant_mode"]
    return Invocation(tensors, (
        OutputSpec((tokens, q_heads, dim), "bfloat16", q.device),
        OutputSpec((q_heads, tokens) if attrs["return_softmax_lse"] else (0,), "float32", q.device),
    ), attrs, inference=None)


OPERATOR = OperatorDefinition(
    name="quant_flash_attn", domain="attention", op=OP, infer=infer,
    schema=(
        "(Tensor q, Tensor k, Tensor v, Tensor q_descale, Tensor k_descale, Tensor v_descale, "
        "*, Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_kv, Tensor metadata, "
        "int quant_mode=5, Tensor? seqused_q=None, Tensor? attn_mask=None, float softmax_scale=1., int mask_mode=0, "
        "int win_left=-1, int win_right=-1, int max_seqlen_q=-1, int max_seqlen_kv=-1, "
        "str layout_q='TND', str layout_q_descale='TND', str layout_kv='PA_BBND', str layout_out='TND', "
        "bool return_softmax_lse=False, int quant_schema=2) -> (Tensor, Tensor)"
    ),
    output_names=("attn_out", "softmax_lse"),
    description="Ascend950 MXFP4 paged GQA with FP32 softmax and AICPU-generated metadata.",
)
