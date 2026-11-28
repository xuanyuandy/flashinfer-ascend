"""TND block-level TopK index expansion for sparse attention pre-processing.

The public contract takes the block-level TopK indices selected by a sparse
attention indexer, ``topk_indices[T1, N2, blockCount]`` INT32 where ``-1``
marks an unused block slot, plus the cumulative TND sequence lengths
``acc_seq_qlen[B+1]`` and ``acc_seq_kvlen[B+1]`` INT64 (prefix sums starting
at 0). It returns ``topk_output[T1, N2, blockCount * blockSize]`` INT32: for
every query token and head each selected block index ``idx`` is expanded to
the ``block_size`` consecutive token indices ``idx * block_size + i`` for
``i in [0, block_size)``.

Positions beyond the causally valid KV range of the token (the flat offset
``s1_idx + (cur_kv - cur_q) + 1`` over the block grid) are ``-1``, and the
trailing lanes of the last valid partial block are ``-1`` as well. ``causal``
is accepted for interface parity with the source operator; the TND token
expansion already encodes the causal lower-triangle rule, so both values
follow the same semantics.

``block_size`` is in ``[1, 8]``: the kernel broadcast materializes exactly
eight token lanes per block index. ``blockCount`` is in ``[1, 4096]`` (the
128 KiB broadcast buffer). The source kernel and reference keep the whole
valid range inside ``blockCount``; callers that select fewer blocks than the
causal valid count are outside the declared contract.

Migrated from gitcode ops-transformer
``experimental/attention/block_indexer_rearrange`` (Atlas A2/A3); this
implementation declares the Ascend910B and Ascend910C targets. Eager
AOT-only and not graph capturable.
"""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.block_indexer_rearrange"
# The kernel broadcast materializes 8 token lanes per block index.
MAX_BLOCK_SIZE = 8
# The kernel broadcast buffer is 128 KiB of INT32 lanes (blockCount * 8 * 4).
MAX_BLOCK_COUNT = 4096
MIN_TOKENS = 1
MAX_TOKENS = 1 << 20
MAX_HEAD_NUM = 1 << 16

INPUT_NAMES = ("topk_indices", "acc_seq_qlen", "acc_seq_kvlen")


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def infer(arguments, layouts):
    """Validate metadata only; never read Tensor values or allocate outputs."""
    topk_indices, acc_seq_qlen, acc_seq_kvlen = (
        arguments[name] for name in INPUT_NAMES
    )
    causal = arguments.get("causal", False)
    if type(causal) is not bool:
        raise ValueError("causal must be a bool")
    block_size = arguments.get("block_size", 1)
    if type(block_size) is not int or not 1 <= block_size <= MAX_BLOCK_SIZE:
        raise ValueError("block_size must be an integer in [1, 8]")

    topk_shape = _shape(topk_indices)
    if len(topk_shape) != 3:
        raise ValueError("topk_indices must be rank-3 [T1, N2, blockCount]")
    if _dtype(topk_indices) != "int32":
        raise ValueError("topk_indices must be int32")
    t1, n2, block_count = topk_shape
    if not MIN_TOKENS <= t1 <= MAX_TOKENS:
        raise ValueError("T1 must be in [1, 1048576]")
    if not 1 <= n2 <= MAX_HEAD_NUM:
        raise ValueError("N2 must be in [1, 65536]")
    if not 1 <= block_count <= MAX_BLOCK_COUNT:
        raise ValueError("blockCount must be in [1, 4096]")

    lengths = (("acc_seq_qlen", acc_seq_qlen), ("acc_seq_kvlen", acc_seq_kvlen))
    for name, tensor in lengths:
        if len(_shape(tensor)) != 1:
            raise ValueError(f"{name} must be rank-1 cumulative [B+1]")
        if _dtype(tensor) != "int64":
            raise ValueError(f"{name} must be int64")
        if _shape(tensor)[0] < 2:
            raise ValueError(f"{name} must be a cumulative [B+1] prefix with B >= 1")
    if layouts is not None:
        for name, _ in lengths + (("topk_indices", topk_indices),):
            if name in layouts and layouts[name] != "ND":
                raise ValueError(f"{name} requires ND physical layout")
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (
            OutputSpec((t1, n2, block_count * block_size), "int32", topk_indices.device),
        ),
        {"causal": causal, "block_size": int(block_size)},
        InferenceSpec(compute_dtype="int32", causal=causal),
    )


def validate_values(arguments):
    """Validate CPU-resident values; device tensors are left untouched."""
    topk_indices = arguments["topk_indices"]
    acc_seq_qlen = arguments["acc_seq_qlen"]
    acc_seq_kvlen = arguments["acc_seq_kvlen"]
    for name, tensor in (("acc_seq_qlen", acc_seq_qlen), ("acc_seq_kvlen", acc_seq_kvlen)):
        if getattr(tensor.device, "type", None) != "cpu":
            continue
        prefix = [int(v) for v in tensor.tolist()]
        if prefix[0] != 0 or any(a > b for a, b in zip(prefix, prefix[1:])):
            raise ValueError(f"{name} must be a nondecreasing cumulative prefix starting at 0")
    if (getattr(acc_seq_qlen.device, "type", None) == "cpu"
            and getattr(topk_indices.device, "type", None) == "cpu"
            and int(acc_seq_qlen.tolist()[-1]) != int(topk_indices.shape[0])):
        raise ValueError("acc_seq_qlen must end at the flat token count T1 of topk_indices")
    if getattr(topk_indices.device, "type", None) == "cpu":
        import torch

        if bool((topk_indices < -1).any()):
            raise ValueError(
                "topk_indices must only contain valid block ids >= 0 or the invalid marker -1")


OPERATOR = OperatorDefinition(
    name="block_indexer_rearrange",
    domain="attention",
    op=OP,
    schema=(
        "(Tensor topk_indices, Tensor acc_seq_qlen, Tensor acc_seq_kvlen, "
        "*, bool causal=False, int block_size=1, bool validate_metadata=False) -> Tensor"
    ),
    output_names=("topk_output",),
    infer=infer,
    validate_values=validate_values,
    description=(
        "TND block-level TopK index expansion for sparse attention "
        "pre-processing, migrated from gitcode ops-transformer "
        "experimental/attention/block_indexer_rearrange. Expands each "
        "selected block index of topk_indices[T1,N2,blockCount] (INT32, -1 "
        "marks an unused slot) into block_size consecutive token indices "
        "idx*block_size+i and returns topk_output[T1,N2,blockCount*blockSize] "
        "INT32. Lanes beyond the causally valid KV range of the token "
        "(s1_idx + cur_kv - cur_q + 1 over the block grid) and the trailing "
        "lanes of the last valid partial block are -1. acc_seq_qlen and "
        "acc_seq_kvlen are cumulative [B+1] prefix sums starting at 0; "
        "acc_seq_qlen must end at T1. causal is interface parity only: the "
        "TND expansion already encodes the causal rule. block_size is in "
        "[1, 8] and blockCount in [1, 4096] (kernel broadcast and UB buffer "
        "limits). Eager AOT-only and not graph capturable."
    ),
)
