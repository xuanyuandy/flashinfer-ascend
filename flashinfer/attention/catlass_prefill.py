"""The bounded, fixed-length A2/A3/A5 CATLASS prefill entry.

The internal ``ascendc.single_prefill_catlass`` op uses the generic ACLNN tensor
ABI. The reserved ``attention.*`` domain binds the separate Attention POD ABI;
this standalone eager package does not claim that framework integration.
"""

from ..runtime import build_signature, current_hardware, dispatch
from ..runtime.runner import run_selected as _run_selected


def single_prefill_with_kv_cache_catlass(
    q, k, v, *, causal=False, kv_layout="NHD", custom_mask=None, out=None,
    backend="auto", kernel_id=None,
):
    """Compute dense ``softmax(q @ k.T / sqrt(128)) @ v`` on A2/A3/A5.

    Inputs are contiguous FP16 or BF16 NPU ND tensors sharing one dtype: ``q[S,H,128]`` and
    ``k[T,H,128]``, ``v[T,H,128]``. S and T are multiples of 128 in
    [128, 512]; H is in [1, 8], shared by Q/K/V. The implicit batch is 1.
    Only NHD, noncausal, unmasked attention is supported. Sequence lengths
    come from complete dense shapes; runtime length tensors are not accepted.
    Optional ``out`` has Q's shape and dtype; output/input aliasing is
    rejected at dispatch before the lazy package runner resolves.

    Unsupported metadata/options fail at dispatch before output allocation.
    Installed AOT packages use the common verified loader; an absent package
    can compile only under the existing JIT policy. This eager entry is not
    graph capturable. It does not claim support for data-dependent tiling.
    """
    import torch
    from ..aclnn._ops import ensure_registered
    from ..runtime import Backend

    ensure_registered()
    options = dict(causal=causal, kv_layout=kv_layout, custom_mask=custom_mask,
                   backend=backend.value if isinstance(backend, Backend) else backend,
                   kernel_id=kernel_id)
    if out is None:
        return torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, **options)
    torch.ops.flashinfer.single_prefill_with_kv_cache_catlass_out(q, k, v, out, **options)
    return out


def _single_prefill_impl(
    q, k, v, *, causal=False, kv_layout="NHD", custom_mask=None, out=None,
    backend="auto", kernel_id=None,
):
    """PrivateUse1 implementation; public calls enter through Torch schemas."""
    import torch

    signature, output_metadata = prefill_signature(
        q, k, v, causal=causal, kv_layout=kv_layout, custom_mask=custom_mask, out=out,
    )
    decision = dispatch(signature, current_hardware(), backend=backend, kernel_id=kernel_id)
    if out is None:
        out = torch.empty(output_metadata.shape, dtype=output_metadata.dtype, device=q.device)
    _run_selected(decision.kernel_id, q, k, v, out)
    return out


def prefill_signature(q, k, v, *, causal=False, kv_layout="NHD", custom_mask=None, out=None):
    """Build the identical validated workload for dispatch and offline tuning."""
    from dataclasses import replace
    from types import SimpleNamespace
    from ..operators.attention.single_prefill_catlass.operator import metadata_reason

    output_metadata = out if out is not None else SimpleNamespace(
        shape=tuple(q.shape), dtype=q.dtype, device=q.device,
        is_contiguous=lambda: True,
    )
    tensors = (q, k, v, output_metadata)
    contract = metadata_reason(q, k, v, output_metadata)
    if out is not None and contract == "supported":
        from ..jit.tensor_adapter import _reject_output_overlap

        from .._matmul import overlap_parameters
        try:
            _reject_output_overlap(overlap_parameters(4), tensors)
        except ValueError as error:
            contract = str(error)
    if causal is not False:
        contract = "catlass prefill supports only causal=False (unmasked attention)"
    elif kv_layout != "NHD":
        contract = "catlass prefill supports only kv_layout='NHD'"
    elif custom_mask is not None:
        contract = "catlass prefill does not support custom_mask"
    signature = build_signature(
        "ascendc.single_prefill_catlass", tensors if out is not None else tensors[:3],
        specializations={"tensor_contract": contract,
                         **({"tensor_contiguity": "0"} if any(not t.is_contiguous() for t in tensors) else {})},
    )
    if out is None:
        signature = replace(
            signature, dtypes=signature.dtypes + (str(output_metadata.dtype).removeprefix("torch."),),
            shapes=signature.shapes + (output_metadata.shape,),
            layouts=signature.layouts + ("",),
            physical_layouts=tuple(
                signature.physical_layouts[index]
                if index < len(signature.physical_layouts) else ""
                for index in range(3)
            ) + ("ND",),
        )
    return signature, output_metadata


__all__ = ["single_prefill_with_kv_cache_catlass"]
