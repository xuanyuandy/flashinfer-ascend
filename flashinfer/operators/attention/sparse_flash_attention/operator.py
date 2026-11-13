"""MLA sparse attention semantics shared by backend implementations.

Indices address KV blocks; -1 terminates each row. Mode 3 uses a bottom-right
causal mask. RoPE inputs contain already encoded positional components.
"""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.sparse_flash_attention"
INT64_MAX = 2**63 - 1
INPUT_NAMES = (
    "query", "key", "value", "sparse_indices", "block_table",
    "actual_seq_lengths_query", "actual_seq_lengths_kv", "query_rope", "key_rope",
)
ATTRIBUTE_NAMES = (
    "scale_value", "sparse_block_size", "layout_query", "layout_kv", "sparse_mode",
    "pre_tokens", "next_tokens", "attention_mode", "return_softmax_lse",
)


def infer(arguments, layouts):
    q, k, v, indices, table, lengths_q, lengths_kv, qr, kr = (
        arguments[name] for name in INPUT_NAMES
    )
    attrs = {name: arguments[name] for name in ATTRIBUTE_NAMES}
    lq, lk = attrs["layout_query"], attrs["layout_kv"]
    if (lq, lk) not in (("BSND", "BSND"), ("TND", "TND"),
                        ("BSND", "PA_BSND"), ("TND", "PA_BSND")):
        raise ValueError("layouts require BSND/BSND, TND/TND or BSND|TND/PA_BSND")
    if len(q.shape) != (4 if lq == "BSND" else 3) or len(k.shape) != (3 if lk == "TND" else 4):
        raise ValueError("query/key ranks do not match their layouts")
    if any(int(d) <= 0 for t in (q, k, v, indices) for d in t.shape):
        raise ValueError("query, key, value and sparse_indices require positive dimensions")
    if q.shape[-1] != 512 or k.shape[-2:] != (1, 512) or tuple(v.shape) != tuple(k.shape):
        raise ValueError("MLA requires query D=512 and matching key/value with one KV head and D=512")
    if not (1 <= q.shape[-2] <= 64 or q.shape[-2] == 128):
        raise ValueError("MLA query heads must be in [1,64] or equal 128")
    dtype = str(q.dtype).removeprefix("torch.")
    if dtype not in ("float16", "bfloat16") or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError("query/key/value must share float16 or bfloat16 dtype")
    if tuple(indices.shape[:-2]) != tuple(q.shape[:-2]) or indices.shape[-2] != 1:
        raise ValueError("sparse_indices must have query token axes, one KV head, and a sparse-block axis")
    if len(indices.shape) != len(q.shape) or str(indices.dtype).removeprefix("torch.") != "int32":
        raise ValueError("sparse_indices requires int32 and the same rank as query")
    if lq == "TND" and lengths_q is None:
        raise ValueError("TND query requires actual_seq_lengths_query prefix sums")
    batch = q.shape[0] if lq == "BSND" else lengths_q.shape[0]
    for name, tensor in (("actual_seq_lengths_query", lengths_q), ("actual_seq_lengths_kv", lengths_kv)):
        if tensor is not None and (tuple(tensor.shape) != (batch,) or str(tensor.dtype).removeprefix("torch.") != "int32"):
            raise ValueError(name + " requires int32 [batch]")
    if lk == "BSND" and k.shape[0] != batch:
        raise ValueError("dense key/value batch differs from query")
    if lk in ("TND", "PA_BSND") and lengths_kv is None:
        raise ValueError(lk + " requires actual_seq_lengths_kv")
    if lk == "PA_BSND":
        if (table is None or len(table.shape) != 2 or table.shape[0] != batch
                or table.shape[1] < 1 or str(table.dtype).removeprefix("torch.") != "int32"):
            raise ValueError("paged KV requires int32 block_table [batch,max_blocks]")
        if not 16 <= k.shape[1] <= 1024 or k.shape[1] % 16:
            raise ValueError("paged KV block size must be a multiple of 16 in [16,1024]")
    elif table is not None:
        raise ValueError("block_table is only valid for PA_BSND")
    if (qr is None) != (kr is None):
        raise ValueError("query_rope and key_rope must be supplied together")
    if qr is not None and (
        tuple(qr.shape) != tuple(q.shape[:-1]) + (64,)
        or tuple(kr.shape) != tuple(k.shape[:-1]) + (64,)
        or qr.dtype != q.dtype or kr.dtype != q.dtype
    ):
        raise ValueError("RoPE tensors must match query/key token axes and dtype with Dr=64")
    block = attrs["sparse_block_size"]
    if type(block) is not int or not 1 <= block <= 128 or block & (block - 1):
        raise ValueError("sparse_block_size must be a power of two in [1,128]")
    if lk == "PA_BSND" and k.shape[1] % block:
        raise ValueError("paged KV block size must be divisible by sparse_block_size")
    if type(attrs["sparse_mode"]) is not int or attrs["sparse_mode"] not in (0, 3):
        raise ValueError("sparse_mode must be 0 (full) or 3 (bottom-right causal)")
    if any(type(attrs[name]) is not int or attrs[name] != INT64_MAX for name in ("pre_tokens", "next_tokens")):
        raise ValueError("pre_tokens and next_tokens currently require INT64_MAX")
    if type(attrs["attention_mode"]) is not int or attrs["attention_mode"] != 2:
        raise ValueError("attention_mode must be 2 (MLA absorb)")
    if type(attrs["return_softmax_lse"]) is not bool:
        raise ValueError("return_softmax_lse must be bool")
    scale = attrs["scale_value"]
    if type(scale) not in (float, int) or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("scale_value must be finite and representable as float32")
    if layouts is not None and any(value != "ND" for value in layouts.values()):
        raise ValueError("SparseFlashAttention requires contiguous ND tensors")
    aux = ((q.shape[0], 1, q.shape[1], q.shape[2]) if lq == "BSND" else (1, q.shape[0], q.shape[1]))
    if not attrs["return_softmax_lse"]:
        aux = (0,)
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (OutputSpec(tuple(q.shape), dtype, q.device),
         OutputSpec(aux, "float32", q.device), OutputSpec(aux, "float32", q.device)),
        attrs,
        InferenceSpec(compute_dtype=dtype, accumulator_dtype="float32",
                      attention_variant="mla", sparse_format="block_sparse",
                      causal=attrs["sparse_mode"] == 3,
                      kv_cache_type={"BSND": "dense", "TND": "ragged", "PA_BSND": "paged"}[lk],
                      auxiliary_outputs=("softmax_max", "softmax_sum") if attrs["return_softmax_lse"] else ()),
    )


def sequence_lengths(arguments):
    """Read length values only during eager execution, never during fake/dispatch."""
    q, k = arguments["query"], arguments["key"]
    lq, lk = arguments["layout_query"], arguments["layout_kv"]
    batch = q.shape[0] if lq == "BSND" else arguments["actual_seq_lengths_query"].shape[0]
    result = []
    for name, layout, total in (("actual_seq_lengths_query", lq, q.shape[0] if lq == "TND" else q.shape[1]),
                               ("actual_seq_lengths_kv", lk, k.shape[0] if lk == "TND" else k.shape[1])):
        tensor = arguments[name]
        values = [total] * batch if tensor is None else tensor.cpu().tolist()
        if layout == "TND":
            if values[-1] != total or any(a > b for a, b in zip([0] + values, values)):
                raise ValueError(name + " prefix sums must be nondecreasing, nonnegative and end at total tokens")
            values = [b - a for a, b in zip([0] + values, values)]
        capacity = arguments["block_table"].shape[1] * k.shape[1] if layout == "PA_BSND" else total
        if any(value < 0 or value > capacity for value in values):
            raise ValueError(name + " values exceed the storage capacity")
        result.append(values)
    return tuple(result)


def validate_values(arguments):
    """Reject invalid device addresses before the native kernel can read them."""
    qlens, klens = sequence_lengths(arguments)
    q, indices = arguments["query"], arguments["sparse_indices"].cpu()
    block = arguments["sparse_block_size"]
    offset = 0
    for batch, (qlen, klen) in enumerate(zip(qlens, klens)):
        rows = indices[batch, :qlen, 0] if arguments["layout_query"] == "BSND" else indices[offset:offset + qlen, 0]
        offset += qlen
        # All rows are checked, including causal-masked selections. Addressing
        # out of range must never depend on a kernel's incidental masking order.
        if bool(((rows < -1) | (rows >= (klen + block - 1) // block)).any()):
            raise ValueError("sparse_indices contains an out-of-range KV block")
        if rows.shape[-1] > 1 and bool(((rows[:, :-1] == -1) & (rows[:, 1:] != -1)).any()):
            raise ValueError("sparse_indices must place all -1 padding after valid indices")
        table = arguments["block_table"]
        if table is not None:
            used = table[batch, :(klen + arguments["key"].shape[1] - 1) // arguments["key"].shape[1]].cpu()
            if bool(((used < 0) | (used >= arguments["key"].shape[0])).any()):
                raise ValueError("block_table contains an out-of-range physical KV block")


OPERATOR = OperatorDefinition(
    name="sparse_flash_attention", domain="attention", op=OP,
    schema="(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, *, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, int sparse_block_size=1, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int attention_mode=2, bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)",
    output_names=("attention_out", "softmax_max", "softmax_sum"),
    infer=infer, validate_values=validate_values,
    description="MLA sparse attention with optional encoded RoPE and paged KV. Returns attention_out, softmax_max and softmax_sum; the latter two are empty unless return_softmax_lse=True. Mode 3 is bottom-right causal. Not graph capturable.",
)
