"""CPU golden and samples for the experimental arch22 lightning indexer.

The reference replicates the gitcode ops-transformer experimental test golden:
head-weighted ReLU(QK) reduction over query heads, causal lower-triangle
masking (sparse_mode=3), sink (init_num) and local window (local_num)
sentinels, optional block aggregation (kv_block_len), then stable TopK.
Indices compare exactly; values are zero unless return_value=True.
"""

import math

DEFAULT_SCORE = -8888888
# Kernel sentinels (lightning_indexer_vector.h): diagonal = FLT_MAX (0x7F7FFFFF),
# sink/local window = FLT_MAX/2, valid scores clipped to FLT_MAX/4.
FLT_MAX = 3.4028234663852886e38
DIGONAL_VALUE = FLT_MAX
WINDOW_VALUE = FLT_MAX / 2
CLIP_MAX_LOGITS = FLT_MAX / 4

CASE_PARAMETERS = (
    {
        # Single-token smoke: one query row against a short paged key.
        "name": "tnd_pa_smoke",
        "batch": 1, "seq_q": 16, "heads": 4, "seq_k": 128,
        "block_len": 1, "init_num": 4, "local_num": 32,
        "sparse_count": 128, "sparse_mode": 3, "return_value": False,
        "layout_q": "TND", "layout_k": "PA_BSND", "dtype": "bfloat16", "seed": 3,
    },
    {
        # Block aggregation with a non-trivial kv_block_len.
        "name": "tnd_pa_block4",
        "batch": 1, "seq_q": 64, "heads": 8, "seq_k": 512,
        "block_len": 4, "init_num": 4, "local_num": 32,
        "sparse_count": 128, "sparse_mode": 3, "return_value": False,
        "layout_q": "TND", "layout_k": "PA_BSND", "dtype": "bfloat16", "seed": 7,
    },
    {
        # Ragged (TND/TND) path: cumulative lengths on both sides.
        "name": "tnd_tnd_ragged",
        "batch": 2, "seq_q": 96, "heads": 8, "seq_k": 256,
        "block_len": 1, "init_num": 0, "local_num": 16,
        "sparse_count": 128, "sparse_mode": 3, "return_value": False,
        "layout_q": "TND", "layout_k": "TND", "dtype": "float16", "seed": 11,
    },
    {
        # Source-test anchor at reduced scale: N1=64 heads (production DSA head
        # count), seq_k=1024, sink=4, local=32. Mirrors the gitcode validation
        # configuration (seq_q=1024/heads=64/seq_k>=4096) within CPU-golden budget.
        "name": "tnd_pa_anchor",
        "batch": 1, "seq_q": 256, "heads": 64, "seq_k": 1024,
        "block_len": 1, "init_num": 4, "local_num": 32,
        "sparse_count": 128, "sparse_mode": 3, "return_value": False,
        "layout_q": "TND", "layout_k": "PA_BSND", "dtype": "bfloat16", "seed": 9,
    },
    {
        # Paged path with return_value=True: values are compared this time.
        "name": "tnd_pa_values",
        "batch": 1, "seq_q": 32, "heads": 4, "seq_k": 256,
        "block_len": 1, "init_num": 4, "local_num": 32,
        "sparse_count": 128, "sparse_mode": 3, "return_value": True,
        "layout_q": "TND", "layout_k": "PA_BSND", "dtype": "bfloat16", "seed": 5,
    },
)


def _split_lengths(torch, configuration):
    """Per-batch query/key lengths for the sample inputs."""
    import numpy as np

    rng = np.random.default_rng(configuration["seed"])
    batch = configuration["batch"]
    seq_q = configuration["seq_q"]
    if configuration.get("layout_q") == "BSND":
        # BSND stores a fixed S1 stride per batch; keep uniform lengths so no
        # padding row exists between batches.
        per = max(1, seq_q // batch)
        q_lengths = [per] * batch
    elif batch == 1:
        q_lengths = [seq_q]
    else:
        raw = rng.integers(1, seq_q, size=batch)
        q_lengths = [max(1, int(v)) for v in raw]
        q_lengths[0] = seq_q - sum(q_lengths[1:])
        if q_lengths[0] < 1:
            q_lengths = [max(1, seq_q // batch)] * batch
    if batch == 1:
        k_lengths = [configuration["seq_k"]]
    elif configuration.get("layout_q") == "BSND":
        per_k = max(1, configuration["seq_k"] // batch)
        k_lengths = [per_k] * batch
    else:
        k_lengths = []
        for q in q_lengths:
            extra = int(rng.integers(16, max(17, configuration["seq_k"] // 2)))
            k_lengths.append(min(configuration["seq_k"], q + extra))
    return q_lengths, k_lengths


def make_cpu_inputs(torch, configuration):
    import numpy as np

    cfg = dict(configuration)
    name = cfg["name"]
    layout_q = cfg["layout_q"]
    layout_k = cfg["layout_k"]
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
    heads = cfg["heads"]
    q_lengths, k_lengths = _split_lengths(torch, cfg)
    total_q = sum(q_lengths)
    total_k = sum(k_lengths)
    rng = np.random.default_rng(cfg["seed"])

    query = torch.tensor(rng.uniform(-2, 2, (total_q, heads, 128))).to(dtype)
    key = torch.tensor(rng.uniform(-2, 2, (total_k, 1, 128))).to(dtype)
    weights = torch.tensor(rng.uniform(-1, 1, (total_q, heads))).to(torch.float32)

    # Length semantics follow the QUERY layout: TND query uses cumulative [B+1]
    # prefix sums on both sides; BSND query uses per-sequence [B] lengths.
    # The native ABI declares int64 lengths; the sample follows the contract.
    if layout_q == "TND":
        cur_q = torch.tensor([0] + list(np.cumsum(q_lengths)), dtype=torch.int64)
        cur_k = torch.tensor([0] + list(np.cumsum(k_lengths)), dtype=torch.int64)
    else:
        cur_q = torch.tensor(q_lengths, dtype=torch.int64)
        cur_k = torch.tensor(k_lengths, dtype=torch.int64)

    flat_key = key
    block_table = None
    device_key = key
    if layout_q == "BSND":
        batch = len(q_lengths)
        s1 = q_lengths[0]
        s2 = k_lengths[0]
        query = query.reshape(batch, s1, heads, 128)
        key = key.reshape(batch, s2, 1, 128)
        weights = weights.reshape(batch, s1, heads)
        device_key = key
    if layout_k == "PA_BSND":
        block_size = 128
        max_seq_len = max(k_lengths)
        width = math.ceil(max_seq_len / block_size)
        block_table = torch.arange(0, len(k_lengths) * width, 1, dtype=torch.int32)
        block_table = block_table.reshape(len(k_lengths), width)
        device_key = torch.zeros(len(k_lengths) * width, block_size, 1, 128).to(dtype)
        start = 0
        for row, kv_len in enumerate(k_lengths):
            used = math.ceil(kv_len / block_size)
            for block_idx in range(used):
                src = start + block_idx * block_size
                dst = int(block_table[row, block_idx])
                take = min(block_size, kv_len - block_idx * block_size)
                device_key[dst, :take] = flat_key[start + block_idx * block_size: src + take]
            start += kv_len

    return dict(
        query=query,
        key=device_key,
        flat_key=flat_key,
        weights=weights,
        cur_seq_lengths_query=cur_q,
        cur_seq_lengths_key=cur_k,
        block_table=block_table,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        layout_q=layout_q,
        layout_k=layout_k,
        sparse_count=int(cfg["sparse_count"]),
        sparse_mode=int(cfg["sparse_mode"]),
        block_len=int(cfg["block_len"]),
        init_num=int(cfg["init_num"]),
        local_num=int(cfg["local_num"]),
        q_block_len=1,
        return_value=bool(cfg["return_value"]),
        dtype=dtype,
        name=name,
    )


def golden(torch, tensors):
    """CPU FP32 reference: weighted head reduce + causal mask + sink/local + TopK."""
    query = tensors["query"]
    key = tensors["flat_key"]
    weights = tensors["weights"]
    lengths_q = tensors["cur_seq_lengths_query"]
    lengths_k = tensors["cur_seq_lengths_key"]
    layout_q = tensors["layout_q"]
    layout_k = tensors["layout_k"]
    sparse_count = tensors["sparse_count"]
    block_len = tensors["block_len"]
    init_num = tensors["init_num"]
    local_num = tensors["local_num"]
    q_block_len = tensors["q_block_len"]
    return_value = tensors["return_value"]

    cumulative = layout_q == "TND"
    if cumulative:
        batch = lengths_q.shape[0] - 1
    else:
        batch = lengths_q.shape[0]

    # BSND batches live at a fixed S1/S2 stride per batch; TND/PA flat
    # sequences advance by the actual per-batch lengths.
    q_stride = int(query.shape[1]) if layout_q == "BSND" else 0
    k_stride = int(key.shape[1]) if (layout_q == "BSND" and layout_k == "BSND") else 0

    out_shape = list(query.shape)
    out_shape[-1] = sparse_count
    out_shape[-2] = 1
    n = query.shape[-2]
    d = query.shape[-1]
    out = (torch.zeros(out_shape, dtype=torch.int32).reshape(-1, 1, sparse_count) - 1)
    values_out = torch.zeros(out_shape, dtype=torch.float32).reshape(-1, 1, sparse_count)
    process_q = 0
    process_k = 0
    for batch_id in range(batch):
        if cumulative:
            act_s1 = int(lengths_q[batch_id + 1] - lengths_q[batch_id])
            act_s2 = int(lengths_k[batch_id + 1] - lengths_k[batch_id])
        else:
            act_s1 = int(lengths_q[batch_id])
            act_s2 = int(lengths_k[batch_id])
        if act_s1 < 0:
            break
        if act_s1 == 0:
            if act_s2 == 0:
                act_s1 = 1
                act_s2 = 1
                process_q += q_stride if q_stride else act_s1
                process_k += k_stride if k_stride else act_s2
                continue
            process_q += q_stride if q_stride else 1
            process_k += k_stride if k_stride else act_s2
            continue
        q_start = process_q
        now_q = query.reshape(-1, n, d)[process_q:process_q + act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_k = key.reshape(-1, d)[process_k:process_k + act_s2, :].transpose(0, 1).to(torch.float32)
        now_weights = (weights.reshape(-1, n, 1)[process_q:process_q + act_s1, :, :]
                       .transpose(0, 1).to(torch.float32))
        process_q += q_stride if q_stride else act_s1
        process_k += k_stride if k_stride else act_s2
        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        reduce_out = torch.sum(relu_out * now_weights, dim=0)

        mask = torch.triu(torch.ones([act_s1, act_s2], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        reduce_out = reduce_out.masked_fill(mask.to(torch.bool), DEFAULT_SCORE)

        act_s2_align = (act_s2 + block_len - 1) // block_len * block_len
        padded = torch.zeros([act_s1, act_s2_align], dtype=torch.float32)
        padded[:, :act_s2] = reduce_out
        padded[:, act_s2:] = DEFAULT_SCORE
        padded = padded.view(act_s1, act_s2_align // block_len, block_len)
        padded = torch.sum(padded, dim=-1) / block_len
        padded = torch.clip(padded, max=CLIP_MAX_LOGITS)

        block_init = init_num // block_len
        block_local = local_num // block_len
        padded[:, :block_init] = WINDOW_VALUE
        for s1 in range(act_s1):
            valid_s2_len = act_s2 - act_s1 + s1 + 1
            valid_s2_block = (valid_s2_len + block_len - 1) // block_len
            start_idx = valid_s2_block - block_local if valid_s2_block - block_local > 0 else 0
            padded[s1, start_idx:valid_s2_block] = WINDOW_VALUE
            padded[s1, valid_s2_block - 1] = DIGONAL_VALUE
            padded[s1, valid_s2_block:] = DEFAULT_SCORE

        block_mask = torch.triu(torch.ones([act_s1, act_s2_align], dtype=torch.uint8),
                                diagonal=(act_s2 - act_s1) + 1)
        block_mask = block_mask.view(act_s1, act_s2_align // block_len, block_len)
        block_mask = (torch.sum(block_mask, dim=-1) == block_len).to(torch.uint8)

        sorted_value, sorted_indices = torch.sort(padded, dim=1, descending=True, stable=True)
        sorted_indices = sorted_indices.masked_fill(block_mask.to(torch.bool), -1)
        if q_block_len > 1:
            raise ValueError("golden only supports q_block_len == 1")
        for s1 in range(act_s1):
            valid_s2_len = act_s2 - act_s1 + s1 + 1
            valid_s2_block = (valid_s2_len + block_len - 1) // block_len
            valid_s2_block = valid_s2_block if valid_s2_block < sparse_count else sparse_count
            tmp = sorted_indices[s1, 0].clone()
            sorted_indices[s1, 0] = sorted_indices[s1, valid_s2_block - 1]
            sorted_indices[s1, valid_s2_block - 1] = tmp
            tmp = sorted_value[s1, 0].clone()
            sorted_value[s1, 0] = sorted_value[s1, valid_s2_block - 1]
            sorted_value[s1, valid_s2_block - 1] = tmp

        return_s2 = min(sparse_count, act_s2_align // block_len)
        out[q_start:q_start + act_s1, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        if return_value:
            values_out[q_start:q_start + act_s1, 0, :return_s2] = sorted_value[:, :return_s2]

    out = out.reshape(out_shape)
    values_out = values_out.reshape(out_shape)
    return out, values_out


def sample(torch, torch_npu, **configuration):
    """Framework acceptance sample: device args/options + CPU golden expected."""
    tensors = make_cpu_inputs(torch, configuration or {})
    indices, values = golden(torch, tensors)
    out_dtype = tensors["dtype"]
    expected_values = values.to(out_dtype) if tensors["return_value"] else \
        torch.zeros_like(values).to(out_dtype)

    def to_device(value):
        if not hasattr(value, "device"):
            return value
        tensor = value.contiguous().npu()
        # Keep ND storage for ND-declared tensors (avoid NCHW default on 4D).
        if tensor.dim() >= 4:
            return torch_npu.npu_format_cast(tensor, 2)
        return tensor

    args = (
        to_device(tensors["query"]),
        to_device(tensors["key"]),
        to_device(tensors["weights"]),
    )
    options = dict(
        cur_seq_lengths_query=to_device(tensors["cur_seq_lengths_query"]),
        cur_seq_lengths_key=to_device(tensors["cur_seq_lengths_key"]),
        block_table=None if tensors["block_table"] is None else to_device(tensors["block_table"]),
        layout_query=tensors["layout_q"],
        layout_key=tensors["layout_k"],
        sparse_count=int(tensors["sparse_count"]),
        kv_block_len=int(tensors["block_len"]),
        q_block_len=int(tensors["q_block_len"]),
        init_num=int(tensors["init_num"]),
        local_num=int(tensors["local_num"]),
        sparse_mode=int(tensors["sparse_mode"]),
        return_value=bool(tensors["return_value"]),
        backend="ascendc",
    )
    return args, options, (indices, expected_values)


TOLERANCES = ((0.0, 0.0), (0.05, 0.05))


def tolerances(configuration):
    if configuration.get("return_value", False):
        return ((0.0, 0.0), (0.05, 0.05))
    return ((0.0, 0.0), (0.0, 0.0))
