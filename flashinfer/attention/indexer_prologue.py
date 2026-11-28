"""Rich IndexerPrologue entry points that are not schema-generated.

The weight preparation helper is a plain, explicit TorchNPU call. It owns no
cache and is never consulted by a forward; callers decide when to run it.
"""

DIM_Q = 32 * 128
DIM_QR = 1280


def prepare_indexer_prologue_qw_weight(wqb):
    """Dynamically quantize a BF16 weight and publish its INT8 NZ/scale pair.

    ``wqb`` is the public BF16 ``[4096,1280]`` weight. The returned NZ tensor is
    the transposed logical ``[1280,4096]`` INT8 weight in physical
    ``FRACTAL_NZ``; the returned FP32 ``[4096]`` scale is the per-channel
    DynamicQuant scale. No result is cached or reused implicitly.
    """
    import torch
    import torch_npu

    if not isinstance(wqb, torch.Tensor):
        raise TypeError("wqb must be a torch.Tensor")
    if str(wqb.dtype).removeprefix("torch.") != "bfloat16":
        raise ValueError("wqb must be bfloat16")
    if tuple(wqb.shape) != (DIM_Q, DIM_QR):
        raise ValueError("wqb must have shape (4096, 1280)")
    if wqb.device.type != torch._C._get_privateuse1_backend_name():
        raise ValueError("wqb must be an NPU tensor")
    if not wqb.is_contiguous() or torch_npu.get_npu_format(wqb) != 2:
        raise ValueError("wqb must be contiguous physical ND")
    weight_q, weight_scale = torch_npu.npu_dynamic_quant(wqb, dst_type=torch.int8)
    weight_nz = torch_npu.npu_format_cast(weight_q.t().contiguous(), 29)
    if torch_npu.get_npu_format(weight_nz) != 29:
        raise RuntimeError("weight preparation did not produce FRACTAL_NZ")
    return weight_nz, weight_scale


K_CACHE_LAST_DIM = 128
K_ROPE_DIM = 64


def _dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _validate_cache(torch, tensor, name, dtype, last_dim):
    if _dtype_name(tensor) != dtype:
        raise ValueError("%s must be %s" % (name, dtype))
    if len(tensor.shape) != 4 or int(tensor.shape[2]) != 1 or int(tensor.shape[3]) != last_dim:
        raise ValueError("%s must have shape [NB,NS,1,%d]" % (name, last_dim))
    strides = tuple(int(value) for value in tensor.stride())
    if strides[-1] != 1:
        raise ValueError("%s requires a contiguous last axis" % name)
    if any(value < 0 for value in strides):
        raise ValueError("%s strides must be nonnegative" % name)
    for size, stride in zip(tensor.shape, strides):
        if int(size) > 1 and stride <= 0:
            raise ValueError(
                "%s requires positive strides on non-singleton dimensions" % name)
    blocks, slots = int(tensor.shape[0]), int(tensor.shape[1])
    if blocks * slots > 1:
        block_stride, slot_stride = strides[0], strides[1]
        slot_span = int(tensor.shape[3])
        # Positive strides do not imply disjoint slots. Accept the two
        # block-major/column-major families whose slot intervals are provably
        # disjoint; reject interleaved geometries that would race across AIVs.
        block_major = ((slots == 1 or slot_stride >= slot_span) and
                       (blocks == 1 or block_stride >= (slots - 1) * slot_stride + slot_span))
        slot_major = ((blocks == 1 or block_stride >= slot_span) and
                      (slots == 1 or slot_stride >= (blocks - 1) * block_stride + slot_span))
        if not (block_major or slot_major):
            raise ValueError(
                "%s block/slot strides overlap distinct physical slots" % name)
    span = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, strides):
        if int(size) > 1:
            span += (int(size) - 1) * stride
    if span >= int(tensor.untyped_storage().nbytes()) // tensor.element_size():
        raise ValueError("%s exceeds its storage span" % name)
    return int(tensor.shape[0]), int(tensor.shape[1])


def _ensure_nd(tensor, name):
    """Relabel an already-ND physical Tensor as ND without copying storage."""
    import torch_npu

    if torch_npu.get_npu_format(tensor) == 2:
        return tensor
    if not tensor.is_contiguous():
        raise ValueError("%s must be physical ND (no relabel without a copy)" % name)
    converted = torch_npu.npu_format_cast(tensor, 2)
    if (torch_npu.get_npu_format(converted) != 2
            or converted.data_ptr() != tensor.data_ptr()
            or converted.untyped_storage().data_ptr()
            != tensor.untyped_storage().data_ptr()):
        raise ValueError("%s format relabel would copy storage" % name)
    return converted


def validate_indexer_prologue_k_static(
        latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache,
        k_scale_cache, norm_eps, *, check_device=True):
    """Static shape/dtype/stride contract; never reads slot values."""
    import math
    import torch

    tensors = (latent, wk, norm_weight, rope_cos, rope_sin, indices,
               k_cache, k_scale_cache)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("IndexerPrologueK requires torch.Tensor arguments")
    if check_device:
        backend = torch._C._get_privateuse1_backend_name()
        if any(tensor.device.type != backend for tensor in tensors):
            raise ValueError("IndexerPrologueK requires NPU Tensors")
        if any(tensor.device != latent.device for tensor in tensors):
            raise ValueError("IndexerPrologueK Tensors must share one device")
    if len(latent.shape) != 2 or int(latent.shape[1]) != 512:
        raise ValueError("latent must be [T,512]")
    tokens = int(latent.shape[0])
    if _dtype_name(latent) != "bfloat16" or _dtype_name(wk) != "bfloat16":
        raise ValueError("latent and wk must be bfloat16")
    if tuple(wk.shape) != (128, 512):
        raise ValueError("wk must be [128,512]")
    if _dtype_name(norm_weight) not in ("bfloat16", "float32"):
        raise ValueError("norm_weight must be bfloat16 or float32")
    if tuple(norm_weight.shape) != (128,):
        raise ValueError("norm_weight must contain 128 elements")
    if _dtype_name(rope_cos) != "float32" or _dtype_name(rope_sin) != "float32":
        raise ValueError("rope_cos and rope_sin must be float32")
    if tuple(rope_cos.shape) != (tokens, 1, 1, K_ROPE_DIM):
        raise ValueError("rope_cos must be [T,1,1,64]")
    if tuple(rope_sin.shape) != (tokens, 1, 1, K_ROPE_DIM):
        raise ValueError("rope_sin must be [T,1,1,64]")
    if _dtype_name(indices) != "int32" or tuple(indices.shape) != (tokens, 2):
        raise ValueError("indices must be int32 [T,2]")
    blocks, slots = _validate_cache(torch, k_cache, "k_cache", "int8", K_CACHE_LAST_DIM)
    scale_blocks, scale_slots = _validate_cache(
        torch, k_scale_cache, "k_scale_cache", "float16", 1)
    if (blocks, slots) != (scale_blocks, scale_slots):
        raise ValueError("k_cache and k_scale_cache must share [NB,NS]")
    if isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float)):
        raise ValueError("norm_eps must be a finite positive float")
    import struct
    value = float(norm_eps)
    narrowed = struct.unpack("<f", struct.pack("<f", value))[0]
    if not math.isfinite(value) or value <= 0 or not math.isfinite(narrowed) or narrowed <= 0:
        raise ValueError(
            "norm_eps must remain finite and positive after binary32 narrowing")
    return tokens, blocks, slots


def _validate_slots(indices, blocks, slots):
    values = indices.detach().cpu().tolist()
    for row, pair in enumerate(values):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("indices row %d must contain [block_id,block_offset]" % row)
        block, slot = int(pair[0]), int(pair[1])
        if block == -1 and slot == -1:
            continue
        if block < 0 or slot < 0:
            raise ValueError("indices row %d has an unsupported negative address" % row)
        if block >= blocks or slot >= slots:
            raise ValueError("indices row %d is outside [NB,NS]" % row)


def validate_indexer_prologue_k_call(
        latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache,
        k_scale_cache, norm_eps):
    """Shared eager/meta/direct contract for the rich K entry.

    Returns None for T=0 after static checks; otherwise validates the slot
    values before any AOT load or native dispatch.
    """
    tokens, blocks, slots = validate_indexer_prologue_k_static(
        latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache,
        k_scale_cache, norm_eps)
    if tokens == 0:
        return None
    import torch_npu
    if torch_npu.npu.is_current_stream_capturing():
        raise RuntimeError(
            "IndexerPrologueK graph_capability=NOT_CAPTURABLE; host slot validation "
            "must be rejected before slot values are read")
    _validate_slots(indices, blocks, slots)
    return tokens


def indexer_prologue_k(
        latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache,
        k_scale_cache, norm_eps=1e-6, kernel_id=None, backend=None):
    """Mutate caller-owned K/scale caches in place; return None.

    ``T=0`` performs only static checks and returns without touching the AOT
    package or reading slot values. Padding is exactly ``[-1,-1]``. Independent
    block/slot strides and exact cache aliases are preserved; duplicate valid
    slots are outside the supported input domain and are not checked here.
    """
    import torch

    # Generic harnesses pass routing hints to every public facade. This rich
    # mutation facade owns its dispatch (the registered Torch schema), so the
    # hints are accepted for ABI compatibility and intentionally unused.
    del kernel_id, backend
    tokens = validate_indexer_prologue_k_static(
        latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache,
        k_scale_cache, norm_eps)[0]
    if tokens == 0:
        return None
    from flashinfer.aclnn._ops import ensure_registered
    ensure_registered()
    # The direct dispatch repeats the static/T0/graph/slot contract and owns
    # the no-copy ND relabel, so both entry paths cannot drift.
    torch.ops.flashinfer.indexer_prologue_k(
        latent, wk, norm_weight, rope_cos, rope_sin, indices,
        k_cache, k_scale_cache, float(norm_eps))
    return None
