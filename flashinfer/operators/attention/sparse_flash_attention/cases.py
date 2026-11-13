"""Independent CPU oracle and deterministic A2 acceptance inputs."""


def golden(torch, query, key, value, indices, scale, *, layout_query="BSND", layout_kv="BSND",
           block_table=None, q_lengths=None, kv_lengths=None, query_rope=None, key_rope=None,
           sparse_block_size=1, sparse_mode=3, return_softmax_lse=False):
    batch = query.shape[0] if layout_query == "BSND" else len(q_lengths)
    q_lengths = q_lengths or [query.shape[1]] * batch
    kv_lengths = kv_lengths or [key.shape[1]] * batch
    output = torch.zeros_like(query)
    aux_shape = ((batch, 1, query.shape[1], query.shape[-2]) if layout_query == "BSND"
                 else (1, query.shape[0], query.shape[-2]))
    maximum = torch.zeros(aux_shape, dtype=torch.float64)
    summation = torch.zeros_like(maximum)
    qo = ko = 0
    for b, (qlen, klen) in enumerate(zip(q_lengths, kv_lengths)):
        q = query[b, :qlen] if layout_query == "BSND" else query[qo:qo + qlen]
        idx = indices[b, :qlen, 0] if layout_query == "BSND" else indices[qo:qo + qlen, 0]
        qr = None if query_rope is None else (query_rope[b, :qlen] if layout_query == "BSND" else query_rope[qo:qo + qlen])
        def unpack(tensor):
            if layout_kv == "PA_BSND":
                blocks = block_table[b, :(klen + key.shape[1] - 1) // key.shape[1]].long()
                return tensor[blocks].reshape(-1, 1, tensor.shape[-1])[:klen, 0]
            return tensor[b, :klen, 0] if layout_kv == "BSND" else tensor[ko:ko + klen, 0]
        k, v = unpack(key).double(), unpack(value).double()
        kr = None if key_rope is None else unpack(key_rope).double()
        for row in range(qlen):
            selected = [block * sparse_block_size + inner for block in idx[row].tolist() if block >= 0
                        for inner in range(sparse_block_size)
                        if block * sparse_block_size + inner < klen
                        and (sparse_mode == 0 or block * sparse_block_size + inner <= klen - qlen + row)]
            if not selected:
                continue
            scores = q[row].double() @ k[selected].T
            if qr is not None:
                scores += qr[row].double() @ kr[selected].T
            scores *= scale
            max_row = scores.amax(dim=-1)
            exponent = (scores - max_row[:, None]).exp()
            sum_row = exponent.sum(dim=-1)
            values = ((exponent / sum_row[:, None]) @ v[selected]).to(query.dtype)
            if layout_query == "BSND":
                output[b, row], maximum[b, 0, row], summation[b, 0, row] = values, max_row, sum_row
            else:
                output[qo + row], maximum[0, qo + row], summation[0, qo + row] = values, max_row, sum_row
        qo += qlen
        ko += klen
    if not return_softmax_lse:
        maximum = summation = torch.empty(0)
    return output, maximum.float(), summation.float()


def sample(torch, torch_npu, *, dtype="float16", layout="BSND", paged=False, heads=4,
           qlen=5, klen=19, block=1, sparse_count=8, causal=True, rope=True, lse=True,
           empty=False, lengths=False, shared_kv=False):
    rng = torch.Generator().manual_seed(20261113 + heads + block + klen)
    dtype = getattr(torch, dtype)
    q_lengths = [qlen, qlen + 2] if layout == "TND" else [qlen, qlen]
    kv_lengths = [klen, klen + 3] if layout == "TND" else [klen, klen]
    def random(shape):
        return (torch.randn(shape, generator=rng) * 0.2).to(dtype)
    def device(tensor):
        # Four-dimensional TorchNPU tensors otherwise default to NCHW.
        return torch_npu.npu_format_cast(tensor.npu(), 2)
    qshape = (2, qlen, heads) if layout == "BSND" else (sum(q_lengths), heads)
    kshape = (2, klen, 1) if layout == "BSND" else (sum(kv_lengths), 1)
    q, k = random(qshape + (512,)), random(kshape + (512,))
    v = random(k.shape)
    qr, kr = (random(qshape + (64,)), random(kshape + (64,))) if rope else (None, None)
    indices = torch.full(qshape[:-1] + (1, sparse_count), -1, dtype=torch.int32)
    for b in range(2):
        block_count = (kv_lengths[b] + block - 1) // block
        chosen = list(range(0, block_count, 2))[:sparse_count - 1]
        if not chosen:
            chosen = [0]
        rows = indices[b] if layout == "BSND" else indices[sum(q_lengths[:b]):sum(q_lengths[:b + 1])]
        if not empty:
            rows[..., :len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    table = None
    layout_kv = layout
    if paged:
        layout_kv = "PA_BSND"
        cache_block = max(16, block)
        blocks_per_batch = (max(kv_lengths) + cache_block - 1) // cache_block
        table = torch.arange(2 * blocks_per_batch - 1, -1, -1, dtype=torch.int32).reshape(2, blocks_per_batch)
        def pack(tensor):
            if tensor is None:
                return None
            packed = torch.zeros(2 * blocks_per_batch, cache_block, 1, tensor.shape[-1], dtype=dtype)
            offset = 0
            for b, count in enumerate(kv_lengths):
                values = tensor[b] if layout == "BSND" else tensor[offset:offset + count]
                for row in range(count):
                    packed[table[b, row // cache_block], row % cache_block] = values[row]
                offset += count
            return packed
        k, v, kr = pack(k), pack(v), pack(kr)
    if shared_kv:
        v = k
    scale = 1 / (576 if rope else 512) ** 0.5
    options = dict(layout_query=layout, layout_kv=layout_kv, sparse_block_size=block,
                   sparse_mode=3 if causal else 0, return_softmax_lse=lse, backend="ascendc")
    if layout == "TND" or lengths:
        values = torch.tensor(q_lengths, dtype=torch.int32)
        options["actual_seq_lengths_query"] = device(values.cumsum(0).int() if layout == "TND" else values)
    if layout == "TND" or paged or lengths:
        values = torch.tensor(kv_lengths, dtype=torch.int32)
        options["actual_seq_lengths_kv"] = device(values.cumsum(0).int() if layout == "TND" and not paged else values)
    if table is not None:
        options["block_table"] = device(table)
    if rope:
        options.update(query_rope=device(qr), key_rope=device(kr))
    expected = golden(torch, q, k, v, indices, scale, layout_query=layout, layout_kv=layout_kv,
                      block_table=table, q_lengths=q_lengths, kv_lengths=kv_lengths, query_rope=qr,
                      key_rope=kr, sparse_block_size=block, sparse_mode=options["sparse_mode"],
                      return_softmax_lse=lse)
    device_key = device(k)
    return (device(q), device_key, device_key if shared_kv else device(v), device(indices), scale), options, expected


CASE_PARAMETERS = (
    dict(),
    dict(dtype="bfloat16", heads=64, causal=False, lengths=True),
    dict(rope=False, lse=False, heads=128),
    dict(layout="TND", block=2, heads=4),
    dict(layout="TND", dtype="bfloat16", block=4, heads=64, causal=False),
    dict(paged=True, lse=False, heads=4),
    dict(layout="TND", paged=True, dtype="bfloat16", block=16, lse=False, heads=64),
    dict(block=128, klen=259, heads=64, causal=False),
    dict(block=2, klen=1100, sparse_count=513, heads=64, qlen=1),
    dict(empty=True, lse=False, heads=64),
    dict(shared_kv=True),
    dict(shared_kv=True, rope=False, dtype="bfloat16", lse=False),
    dict(shared_kv=True, layout="TND", paged=True, lse=False, heads=64),
    dict(shared_kv=True, block=1, klen=1100, sparse_count=513, heads=64, qlen=1),
)
# Bounds cover FP16/BF16 probability rounding in the imported implementation.
TOLERANCES = ((0.006, 0.03), (0.003, 0.01), (0.03, 0.01))
