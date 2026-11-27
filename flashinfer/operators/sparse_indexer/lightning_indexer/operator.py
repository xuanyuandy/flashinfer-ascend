"""Lightning indexer semantics for the experimental Ascend910B/C operator.

Computes per-token head-weighted ReLU(QK^T) scores, reduces them over query
heads, then selects the ``sparse_count`` highest-scoring key blocks per query
token. The arch22 native implementation is migrated from the gitcode
ops-transformer ``experimental/attention/lightning_indexer`` operator; this
module keeps that public interface (``cur_seq_lengths_*``, ``kv_block_len``,
``q_block_len``, ``init_num``, ``local_num``, ``return_value``).

Selection semantics for block selection (``kv_block_len`` > 1) add block score
reduction before TopK; this is the experimental extension over the official
``npu_lightning_indexer`` interface.
"""

from types import SimpleNamespace

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "sparse_indexer.lightning_indexer"
INT64_MAX = 2**63 - 1
HEAD_DIM = 128
MAX_TOPK = 4096
MAX_INIT_NUM = 16
MAX_LOCAL_NUM = 2048
BLOCK_LEN_CHOICES = (1, 2, 4, 8, 16)


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _meta(arguments):
    """Validate shape/metadata facts without reading Tensor values."""
    query = arguments["query"]
    key = arguments["key"]
    weights = arguments["weights"]
    layout_q = str(arguments["layout_query"]).upper()
    layout_k = str(arguments["layout_key"]).upper()
    if layout_q not in ("BSND", "TND"):
        raise ValueError("layout_query must be BSND or TND")
    if layout_k not in ("BSND", "TND", "PA_BSND"):
        raise ValueError("layout_key must be BSND, TND or PA_BSND")
    if layout_k != "PA_BSND" and layout_q != layout_k:
        raise ValueError("non-paged key requires layout_query == layout_key")

    sparse_count = arguments["sparse_count"]
    kv_block_len = arguments["kv_block_len"]
    q_block_len = arguments["q_block_len"]
    init_num = arguments["init_num"]
    local_num = arguments["local_num"]
    sparse_mode = arguments["sparse_mode"]
    if not isinstance(sparse_count, int) or not (1 <= sparse_count <= MAX_TOPK):
        raise ValueError("sparse_count must be in [1, 4096]")
    if sparse_count % 128 != 0:
        raise ValueError("sparse_count must be divisible by 128")
    if kv_block_len not in BLOCK_LEN_CHOICES:
        raise ValueError("kv_block_len must be one of 1/2/4/8/16")
    if q_block_len != 1:
        raise ValueError("q_block_len must be 1")
    if not isinstance(init_num, int) or not (0 <= init_num <= MAX_INIT_NUM) or init_num % kv_block_len:
        raise ValueError("init_num must be in [0, 16] and divisible by kv_block_len")
    if not isinstance(local_num, int) or not (0 <= local_num <= MAX_LOCAL_NUM) or local_num % kv_block_len:
        raise ValueError("local_num must be in [0, 2048] and divisible by kv_block_len")
    if sparse_mode not in (0, 3):
        raise ValueError("sparse_mode only supports 0 or 3")

    qshape = tuple(int(v) for v in query.shape)
    kshape = tuple(int(v) for v in key.shape)
    wshape = tuple(int(v) for v in weights.shape)
    if len(qshape) != (4 if layout_q == "BSND" else 3):
        raise ValueError("query rank does not match layout_query")
    if len(kshape) != (3 if layout_k == "TND" else 4):
        raise ValueError("key rank does not match layout_key")
    if len(wshape) != len(qshape) - 1:
        raise ValueError("weights rank must be one below query rank")
    if qshape[-1] != HEAD_DIM:
        raise ValueError("query head dim must be 128")
    if kshape[-1] != HEAD_DIM:
        raise ValueError("key head dim must be 128")
    n2 = 1
    if layout_k == "TND":
        if kshape[1] != n2:
            raise ValueError("key head num must be 1")
        block_size, block_count = 0, 0
    else:
        if kshape[-2] != n2:
            raise ValueError("key head num must be 1")
        block_size, block_count = kshape[1], kshape[0]
    if layout_k == "PA_BSND" and (block_size % 16 or not 0 < block_size <= 4096):
        raise ValueError("paged key block_size must be a positive multiple of 16 and <= 4096")
    if layout_k == "PA_BSND" and block_count < 1:
        raise ValueError("paged key block count must be positive")
    if _dtype(query) not in ("float16", "bfloat16") or _dtype(key) != _dtype(query):
        raise ValueError("query and key must share float16 or bfloat16 dtype")
    if _dtype(weights) != "float32":
        raise ValueError("weights must be float32")

    lengths_q = arguments["cur_seq_lengths_query"]
    lengths_k = arguments["cur_seq_lengths_key"]
    table = arguments["block_table"]
    if layout_q == "TND":
        if lengths_q is None or tuple(int(v) for v in lengths_q.shape)[0] < 2:
            raise ValueError("TND query requires cumulative cur_seq_lengths_query of shape [B+1]")
        if _dtype(lengths_q) not in ("int32", "int64"):
            raise ValueError("cur_seq_lengths_query must be int32 or int64")
        batch = int(lengths_q.shape[0]) - 1
        total_q = qshape[0]
    else:
        batch, total_q = qshape[0], qshape[0] * qshape[1]
        if lengths_q is not None and (len(tuple(int(v) for v in lengths_q.shape)) != 1 or
                                      int(lengths_q.shape[0]) != batch):
            raise ValueError("BSND cur_seq_lengths_query must have shape [B]")
    if layout_k == "PA_BSND":
        if table is None or len(tuple(int(v) for v in table.shape)) != 2 or int(table.shape[0]) != batch:
            raise ValueError("paged key requires block_table of shape [B, max_blocks]")
        if _dtype(table) != "int32":
            raise ValueError("block_table must be int32")
        if lengths_k is None:
            raise ValueError("paged key requires cur_seq_lengths_key")
    elif layout_k == "TND" and lengths_k is None:
        raise ValueError("TND key requires cur_seq_lengths_key")
    if layout_k != "PA_BSND" and table is not None:
        raise ValueError("block_table is only valid with layout_key=PA_BSND")
    if lengths_k is not None:
        expected = batch + 1 if layout_q == "TND" else batch
        if tuple(int(v) for v in lengths_k.shape) != (expected,):
            raise ValueError("cur_seq_lengths_key must have %d elements for %s query" % (expected, layout_q))
        if _dtype(lengths_k) not in ("int32", "int64"):
            raise ValueError("cur_seq_lengths_key must be int32 or int64")

    if layout_q == "BSND":
        expected_weights = (batch, qshape[1], qshape[2])
        out_shape = (batch, qshape[1], n2, sparse_count)
    else:
        expected_weights = (total_q, qshape[1])
        out_shape = (total_q, n2, sparse_count)
    if wshape != expected_weights:
        raise ValueError("weights shape %s does not match query %s" % (wshape, expected_weights))

    return SimpleNamespace(
        layout_q=layout_q,
        layout_k=layout_k,
        batch=batch,
        n1=qshape[1] if layout_q == "BSND" else qshape[1],
        n2=n2,
        head_dim=HEAD_DIM,
        block_size=block_size,
        block_count=block_count,
        sparse_count=sparse_count,
        kv_block_len=kv_block_len,
        out_shape=out_shape,
        dtype=_dtype(query),
        kv_cache_type={"BSND": "dense", "TND": "ragged", "PA_BSND": "paged"}[layout_k],
        causal=sparse_mode == 3,
    )


def infer(arguments, layouts):
    meta = _meta(arguments)
    if layouts is not None:
        bad = [name for name, layout in layouts.items() if layout != "ND"]
        if bad:
            raise ValueError("lightning_indexer requires contiguous ND tensors (non-ND: %s)" % bad)
    device = arguments["query"].device
    inputs = (
        arguments["query"],
        arguments["key"],
        arguments["weights"],
        arguments["cur_seq_lengths_query"],
        arguments["cur_seq_lengths_key"],
        arguments["block_table"],
    )
    return Invocation(
        inputs,
        (
            OutputSpec(meta.out_shape, "int32", device),
            OutputSpec(meta.out_shape, meta.dtype, device),
        ),
        {
            "layout_query": meta.layout_q,
            "layout_key": meta.layout_k,
            "sparse_count": int(arguments["sparse_count"]),
            "kv_block_len": int(arguments["kv_block_len"]),
            "q_block_len": int(arguments["q_block_len"]),
            "init_num": int(arguments["init_num"]),
            "local_num": int(arguments["local_num"]),
            "sparse_mode": int(arguments["sparse_mode"]),
            "pre_tokens": int(arguments["pre_tokens"]),
            "next_tokens": int(arguments["next_tokens"]),
            "return_value": bool(arguments["return_value"]),
        },
        InferenceSpec(
            compute_dtype=meta.dtype,
            accumulator_dtype="float32",
            attention_variant="lightning_indexer",
            kv_cache_type=meta.kv_cache_type,
            causal=meta.causal,
        ),
    )


_METADATA_TENSOR_NAMES = ("cur_seq_lengths_query", "cur_seq_lengths_key", "block_table")


def _strict_metadata(arguments):
    if arguments.get("validate_metadata", False):
        return True
    for name in _METADATA_TENSOR_NAMES:
        tensor = arguments.get(name)
        if tensor is not None and getattr(tensor.device, "type", None) != "cpu":
            return False
    return True


def validate_values(arguments):
    if not _strict_metadata(arguments):
        return
    meta = _meta(arguments)
    lengths_q = arguments["cur_seq_lengths_query"]
    if meta.layout_q == "TND":
        prefix = lengths_q.cpu().tolist()
        total = meta.out_shape[0]
        if prefix[0] != 0 or prefix[-1] != total or any(a > b for a, b in zip(prefix, prefix[1:])):
            raise ValueError("cur_seq_lengths_query must be nondecreasing, start at 0 and end at the token total")
    lengths_k = arguments["cur_seq_lengths_key"]
    if lengths_k is not None and meta.layout_k == "PA_BSND" and meta.layout_q == "TND":
        prefix = lengths_k.cpu().tolist()
        if prefix[0] != 0 or any(a > b for a, b in zip(prefix, prefix[1:])):
            raise ValueError("cur_seq_lengths_key must be nondecreasing and start at 0")
        width = int(arguments["block_table"].shape[1])
        if any((v + meta.block_size - 1) // meta.block_size > width for v in prefix):
            raise ValueError("block_table width is smaller than cur_seq_lengths_key requires")


OPERATOR = OperatorDefinition(
    name="lightning_indexer",
    domain="sparse_indexer",
    op=OP,
    schema=(
        "(Tensor query, Tensor key, Tensor weights, *, "
        "Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, "
        "Tensor? block_table=None, str layout_query='BSND', str layout_key='PA_BSND', "
        "int sparse_count=2048, int kv_block_len=1, int q_block_len=1, "
        "int init_num=0, int local_num=0, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
        "bool return_value=False, bool validate_metadata=False) -> (Tensor, Tensor)"
    ),
    output_names=("sparse_indices", "sparse_values"),
    infer=infer,
    validate_values=validate_values,
    description=(
        "Experimental arch22 lightning indexer migrated from gitcode "
        "ops-transformer: head-weighted ReLU(QK) reduction followed by TopK "
        "block/token selection with sink (init_num), local window (local_num) "
        "and block aggregation (kv_block_len). Returns (sparse_indices, "
        "sparse_values); values are zero unless return_value=True. "
        "TND query with TND/PA_BSND key only: the source kernel's dense BSND "
        "path is internally inconsistent and is not declared. "
        "Eager AOT-only and not graph capturable."
    ),
)
