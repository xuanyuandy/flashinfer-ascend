"""The bounded, fixed-length Ascend950 CATLASS prefill entry.

The internal ``ascendc.single_prefill_catlass`` op uses the generic ACLNN tensor
ABI. The reserved ``attention.*`` domain binds the separate Attention POD ABI;
this standalone eager package does not claim that framework integration.
"""

from ..runtime import build_signature, current_hardware, dispatch, get_global_registry


def _run_selected(kernel_id, *tensors):
    return get_global_registry().get(kernel_id).runner(*tensors)


def single_prefill_with_kv_cache_catlass(
    q, k, v, *, causal=False, kv_layout="NHD", custom_mask=None, out=None,
    backend="auto", kernel_id=None,
):
    """Compute dense ``softmax(q @ k.T / sqrt(128)) @ v`` on Ascend950.

    Inputs are contiguous FP16 NPU ND tensors: ``q[S,H,128]`` and
    ``k[T,H,128]``, ``v[T,H,128]``. S and T are multiples of 128 in
    [128, 512]; H is in [1, 8], shared by Q/K/V. The implicit batch is 1.
    Only NHD, noncausal, unmasked attention is supported. Sequence lengths
    come from complete dense shapes; runtime length tensors are not accepted.
    Optional ``out`` has Q's shape and FP16 dtype; output/input aliasing is
    rejected at dispatch before the lazy package runner resolves.

    Unsupported metadata/options fail at dispatch before output allocation.
    Installed AOT packages use the common verified loader; an absent package
    can compile only under the existing JIT policy. This eager entry is not
    graph capturable. It does not claim support for data-dependent tiling.
    """
    import torch
    from dataclasses import replace
    from types import SimpleNamespace
    from ..ascendc import FLASH_ATTENTION_KERNEL_ID, load_contract, tensor_contract_specialization

    output_metadata = out if out is not None else SimpleNamespace(
        shape=tuple(q.shape), dtype=torch.float16, device=q.device,
        is_contiguous=lambda: True,
    )
    tensors = (q, k, v, output_metadata)
    contract = tensor_contract_specialization(FLASH_ATTENTION_KERNEL_ID, tensors)
    if out is not None and contract == "supported":
        from ..jit.tensor_adapter import _reject_output_overlap

        _, recipe, _ = load_contract(FLASH_ATTENTION_KERNEL_ID)
        try:
            _reject_output_overlap(recipe.tensor_parameters, tensors)
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
        specializations={"tensor_contract": contract},
    )
    if out is None:
        signature = replace(
            signature, dtypes=signature.dtypes + ("float16",),
            shapes=signature.shapes + (output_metadata.shape,),
            layouts=signature.layouts + ("",),
            physical_layouts=tuple(
                signature.physical_layouts[index]
                if index < len(signature.physical_layouts) else ""
                for index in range(3)
            ) + ("ND",),
        )
    decision = dispatch(signature, current_hardware(), backend=backend, kernel_id=kernel_id)
    if out is None:
        out = torch.empty(output_metadata.shape, dtype=torch.float16, device=q.device)
    _run_selected(decision.kernel_id, q, k, v, out)
    return out


__all__ = ["single_prefill_with_kv_cache_catlass"]
