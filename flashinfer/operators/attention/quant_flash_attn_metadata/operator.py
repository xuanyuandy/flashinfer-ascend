"""Public contract for the C4A4 FA/FD schedule producer."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.operators.attention.quant_flash_attn.operator import check_layouts, check_tensor

# Ascend 950DT has up to 36 Cube and 72 Vector cores. The AICPU uses the
# effective core counts when writing records into this device-independent capacity.
MAX_ASCEND950_CORES = 108
METADATA_STRIDE = 16
METADATA_ALIGNMENT = 4096


def infer(arguments, layouts):
    cu_seqlens_q = arguments["cu_seqlens_q"]
    seqused_q = arguments["seqused_q"]
    seqused_kv = arguments["seqused_kv"]
    num_heads_q = arguments["num_heads_q"]
    num_heads_kv = arguments["num_heads_kv"]
    head_dim = arguments["head_dim"]
    quant_mode = arguments["quant_mode"]
    max_seqlen_q = arguments["max_seqlen_q"]
    max_seqlen_kv = arguments["max_seqlen_kv"]
    mask_mode = arguments["mask_mode"]
    layout_q = arguments["layout_q"]
    layout_q_descale = arguments["layout_q_descale"]
    layout_kv = arguments["layout_kv"]
    layout_out = arguments["layout_out"]

    check_layouts(layout_q, layout_q_descale, layout_kv, layout_out, mask_mode)
    if (quant_mode != 5 or head_dim not in (128, 256) or arguments["head_dim_v"] not in (None, head_dim)
            or not 0 < num_heads_kv <= num_heads_q <= 2**31 - 1 or num_heads_q % num_heads_kv):
        raise ValueError("metadata requires mode 5, matching D/Dv and valid GQA heads")
    if any(length != -1 and not 0 < length <= 2**31 - 1 for length in (max_seqlen_q, max_seqlen_kv)):
        raise ValueError("maximum lengths must be positive or -1")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] < 2:
        raise ValueError("metadata requires a cumulative length Tensor of shape [B+1]")

    batch_size = cu_seqlens_q.shape[0] - 1
    check_tensor(cu_seqlens_q, "cu_seqlens_q", "int32", (batch_size + 1,))
    check_tensor(seqused_kv, "seqused_kv", "int32", (batch_size,))
    if seqused_q is not None:
        check_tensor(seqused_q, "seqused_q", "int32", (batch_size,))
    if layouts is not None and (any(layout != "ND" for layout in layouts.values()) or any(
            not tensor.is_contiguous() for tensor in (cu_seqlens_q, seqused_q, seqused_kv) if tensor is not None)):
        raise ValueError("metadata lengths must be contiguous ND on one device")

    records = 1 + MAX_ASCEND950_CORES * batch_size * num_heads_q
    elements = METADATA_STRIDE * records
    row_size = (elements + METADATA_ALIGNMENT - 1) // METADATA_ALIGNMENT * METADATA_ALIGNMENT
    if row_size > 2**31 - 1:
        raise ValueError("metadata schedule exceeds int32 capacity")

    return Invocation(
        inputs=(cu_seqlens_q, None, seqused_q, seqused_kv),
        outputs=(OutputSpec((2, row_size), "int32", cu_seqlens_q.device),),
        attributes=dict(
            batch_size=batch_size, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv,
            num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, head_dim=head_dim, head_dim_v=head_dim,
            quant_mode=quant_mode, mask_mode=mask_mode, layout_q=layout_q, layout_q_descale=layout_q_descale,
            layout_kv=layout_kv, layout_out=layout_out,
        ),
        inference=None,
    )

OPERATOR = OperatorDefinition(
    name="quant_flash_attn_metadata", domain="attention", op="attention.quant_flash_attn_metadata",
    schema=(
        "(int num_heads_q, int num_heads_kv, int head_dim, int quant_mode=5, "
        "*, Tensor cu_seqlens_q, Tensor seqused_kv, Tensor? seqused_q=None, "
        "int max_seqlen_q=-1, int max_seqlen_kv=-1, int? head_dim_v=None, int mask_mode=0, "
        "str layout_q='TND', str layout_q_descale='TND', str layout_kv='PA_BBND', str layout_out='TND') -> Tensor"
    ),
    output_names=("metadata",),
    infer=infer,
    description="Generate an Ascend950 C4A4 FA/FD schedule with the AICPU.",
)
