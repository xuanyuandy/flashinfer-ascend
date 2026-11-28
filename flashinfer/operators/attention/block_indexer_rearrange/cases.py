"""CPU golden and deterministic TND samples for block_indexer_rearrange.

The reference replicates the gitcode ops-transformer experimental test
golden: every selected block index expands to ``idx * block_size + i`` lanes,
invalid slots stay ``-1``, and lanes beyond the causally valid KV range of
the token (including the tail of the last valid partial block) are ``-1``.
Indices compare exactly. All declared cases keep the per-token valid block
count inside ``blockCount``, the range the source kernel and reference
define; the input generator keeps ``cur_kv >= cur_q`` per batch so every
token has at least one valid block.
"""

import math

from .operator import MAX_BLOCK_COUNT, MAX_BLOCK_SIZE

CASE_PARAMETERS = (
    {
        # Source-test anchor: single batch, causal, bs=4, bc=512.
        "name": "tnd_b1_bs4_bc512",
        "n2": 1, "cur_seqlen_q": (200,), "cur_seqlen_kv": (200,),
        "block_size": 4, "block_count": 512, "causal": True, "seed": 41,
    },
    {
        # Two ragged batches where kv outgrows q.
        "name": "tnd_b2_bs4_bc512",
        "n2": 1, "cur_seqlen_q": (200, 300), "cur_seqlen_kv": (400, 900),
        "block_size": 4, "block_count": 512, "causal": True, "seed": 43,
    },
    {
        # bs=2 with a wider grid.
        "name": "tnd_b2_bs2_bc1024",
        "n2": 1, "cur_seqlen_q": (512, 512), "cur_seqlen_kv": (1024, 2048),
        "block_size": 2, "block_count": 1024, "causal": True, "seed": 47,
    },
    {
        # Source-test largest shape: bs=8, bc=2048, T1=3072.
        "name": "tnd_b2_bs8_bc2048",
        "n2": 1, "cur_seqlen_q": (1024, 2048), "cur_seqlen_kv": (2048, 8192),
        "block_size": 8, "block_count": 2048, "causal": True, "seed": 53,
    },
    {
        # Multi-head ragged batch with bs=1 (pure flat expansion).
        "name": "tnd_b3_bs1_heads",
        "n2": 2, "cur_seqlen_q": (1, 17, 300), "cur_seqlen_kv": (3, 100, 512),
        "block_size": 1, "block_count": 1024, "causal": True, "seed": 59,
    },
    {
        # causal=False follows the same TND expansion (interface parity).
        "name": "tnd_b2_bs8_noncausal",
        "n2": 1, "cur_seqlen_q": (128, 300), "cur_seqlen_kv": (256, 800),
        "block_size": 8, "block_count": 512, "causal": False, "seed": 61,
    },
    {
        # Boundary: the last token's valid block count equals blockCount.
        "name": "tnd_b1_bs4_bc16_full",
        "n2": 1, "cur_seqlen_q": (40,), "cur_seqlen_kv": (64,),
        "block_size": 4, "block_count": 16, "causal": True, "seed": 67,
    },
)


def _check_configuration(configuration):
    n2 = int(configuration.get("n2", 1))
    cur_seqlen_q = tuple(int(v) for v in configuration["cur_seqlen_q"])
    cur_seqlen_kv = tuple(int(v) for v in configuration["cur_seqlen_kv"])
    block_size = int(configuration["block_size"])
    block_count = int(configuration["block_count"])
    if not 1 <= block_size <= MAX_BLOCK_SIZE:
        raise ValueError("block_size must be in [1, 8]")
    if not 1 <= block_count <= MAX_BLOCK_COUNT:
        raise ValueError("block_count must be in [1, 4096]")
    if len(cur_seqlen_q) != len(cur_seqlen_kv):
        raise ValueError("cur_seqlen_q and cur_seqlen_kv must have the same batch count")
    if any(kv < q for q, kv in zip(cur_seqlen_q, cur_seqlen_kv)):
        raise ValueError("every batch requires cur_seqlen_kv >= cur_seqlen_q")
    return n2, cur_seqlen_q, cur_seqlen_kv, block_size, block_count


def _get_tnd_idx(acc_q_len, t_idx):
    b_idx = 0
    while t_idx >= acc_q_len[b_idx + 1]:
        b_idx += 1
    s1_idx = t_idx - acc_q_len[b_idx]
    return b_idx, s1_idx


def _randperm(torch, n, device):
    if n <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    return torch.randperm(n, device=device).to(torch.int32)


def make_cpu_inputs(torch, configuration):
    """Deterministic TND inputs following the source-test generator."""
    n2, cur_seqlen_q, cur_seqlen_kv, block_size, block_count = _check_configuration(configuration)
    seed = int(configuration.get("seed", 42))
    torch.manual_seed(seed)

    acc_q = [0]
    acc_kv = [0]
    for q, kv in zip(cur_seqlen_q, cur_seqlen_kv):
        acc_q.append(acc_q[-1] + q)
        acc_kv.append(acc_kv[-1] + kv)
    t1 = acc_q[-1]

    topk_indices = torch.full((t1, n2, block_count), -1, dtype=torch.int32)
    for t in range(t1):
        b_idx, s1_idx = _get_tnd_idx(acc_q, t)
        threshold = cur_seqlen_kv[b_idx] - cur_seqlen_q[b_idx] + 1 + s1_idx
        valid_block_count = math.ceil(threshold / block_size)
        for j in range(n2):
            if valid_block_count >= block_count:
                topk_indices[t][j][-1] = valid_block_count - 1
                topk_indices[t][j][0:-1] = _randperm(torch, valid_block_count - 1, "cpu")[
                    : block_count - 1
                ]
            else:
                topk_indices[t][j][valid_block_count - 1] = valid_block_count - 1
                topk_indices[t][j][0: valid_block_count - 1] = _randperm(
                    torch, valid_block_count - 1, "cpu"
                )

    return {
        "topk_indices": topk_indices,
        "acc_seq_qlen": torch.tensor(acc_q, dtype=torch.int64),
        "acc_seq_kvlen": torch.tensor(acc_kv, dtype=torch.int64),
        "cur_seqlen_q": cur_seqlen_q,
        "cur_seqlen_kv": cur_seqlen_kv,
        "n2": n2,
        "block_size": block_size,
        "block_count": block_count,
        "causal": bool(configuration.get("causal", False)),
        "name": configuration.get("name", "sample"),
    }


def golden(torch, tensors):
    """CPU oracle: block indices expanded to token indices with tail masking."""
    topk_indices = tensors["topk_indices"]
    acc_q = tensors["acc_seq_qlen"].tolist()
    cur_q = tensors["cur_seqlen_q"]
    cur_kv = tensors["cur_seqlen_kv"]
    block_size = int(tensors["block_size"])
    block_count = int(tensors["block_count"])

    t1, n2, _ = topk_indices.shape
    expanded = topk_indices.unsqueeze(-1) * block_size + torch.arange(
        block_size, dtype=torch.int32
    )
    mask = (topk_indices <= -1).unsqueeze(-1)
    fill_value = torch.full_like(expanded, -1)
    golden = torch.where(mask, fill_value, expanded).view(t1, n2, -1)

    for t in range(t1):
        b_idx, s1_idx = _get_tnd_idx(acc_q, t)
        threshold = cur_kv[b_idx] - cur_q[b_idx] + 1 + s1_idx
        valid_block_count = math.ceil(threshold / block_size)
        mask_size = valid_block_count * block_size - threshold
        if valid_block_count > block_count:
            if mask_size != 0:
                golden[t, :, -mask_size:] = -1
        else:
            if mask_size != 0:
                end_pos = valid_block_count * block_size
                golden[t, :, end_pos - mask_size: end_pos] = -1
    return golden


def naive_reference(torch, tensors):
    """Independent per-position oracle used by the self-consistency tests.

    For each token the flat position ``j * block_size + i`` carries
    ``idx * block_size + i`` when the block index is valid and the position is
    inside the causally valid KV range, and ``-1`` otherwise. The declared
    cases keep the valid block count inside ``blockCount`` and every invalid
    slot outside it, which is exactly the contract the kernel and the
    vectorized golden implement.
    """
    topk_indices = tensors["topk_indices"]
    acc_q = tensors["acc_seq_qlen"].tolist()
    cur_q = tensors["cur_seqlen_q"]
    cur_kv = tensors["cur_seqlen_kv"]
    block_size = int(tensors["block_size"])
    block_count = int(tensors["block_count"])

    t1, n2, _ = topk_indices.shape
    out = torch.full((t1, n2, block_count * block_size), -1, dtype=torch.int32)
    for t in range(t1):
        b_idx, s1_idx = _get_tnd_idx(acc_q, t)
        threshold = cur_kv[b_idx] - cur_q[b_idx] + 1 + s1_idx
        for n in range(n2):
            for j in range(block_count):
                index = int(topk_indices[t, n, j])
                for i in range(block_size):
                    position = j * block_size + i
                    if index < 0 or position >= threshold:
                        out[t, n, position] = -1
                    else:
                        out[t, n, position] = index * block_size + i
    return out


def sample(torch, torch_npu, **configuration):
    """Framework acceptance sample: device args/options + CPU golden expected."""
    tensors = make_cpu_inputs(torch, configuration or {})
    expected = golden(torch, tensors)

    def to_device(value):
        if not hasattr(value, "device"):
            return value
        tensor = value.contiguous().npu()
        # Keep ND storage for ND-declared tensors (avoid NCHW default on 4D).
        if tensor.dim() >= 4:
            return torch_npu.npu_format_cast(tensor, 2)
        return tensor

    args = (
        to_device(tensors["topk_indices"]),
        to_device(tensors["acc_seq_qlen"]),
        to_device(tensors["acc_seq_kvlen"]),
    )
    options = dict(
        causal=bool(tensors["causal"]),
        block_size=int(tensors["block_size"]),
        backend="ascendc",
    )
    return args, options, (expected,)
