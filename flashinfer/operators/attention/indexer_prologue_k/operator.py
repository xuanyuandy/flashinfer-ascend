"""Rich mutation-only declaration for attention.indexer_prologue_k."""

from flashinfer.operators import OperatorDefinition

OP = "attention.indexer_prologue_k"

K_TORCH_SCHEMA = (
    "(Tensor latent, Tensor wk, Tensor norm_weight, Tensor rope_cos, Tensor rope_sin, "
    "Tensor indices, Tensor(a!) k_cache, Tensor(b!) k_scale_cache, "
    "float norm_eps=1e-06) -> ()"
)
K_MUTATES_ARGS = ("k_cache", "k_scale_cache")


def _dispatch(latent, wk, norm_weight, rope_cos, rope_sin, indices,
              k_cache, k_scale_cache, norm_eps=1e-6):
    """Dispatch the exact native 10-Tensor ABI with exact cache aliases."""
    import torch
    from flashinfer.runtime import InferenceSpec, build_signature, current_hardware, dispatch
    from flashinfer.runtime.runner import run_selected
    from flashinfer.attention.indexer_prologue import (
        _ensure_nd,
        validate_indexer_prologue_k_call,
    )

    tokens = validate_indexer_prologue_k_call(
        latent, wk, norm_weight, rope_cos, rope_sin, indices,
        k_cache, k_scale_cache, norm_eps)
    if tokens is None:
        return None
    cos_view = _ensure_nd(rope_cos.view(tokens, 64), "rope_cos")
    sin_view = _ensure_nd(rope_sin.view(tokens, 64), "rope_sin")
    k_cache = _ensure_nd(k_cache, "k_cache")
    k_scale_cache = _ensure_nd(k_scale_cache, "k_scale_cache")
    tensors = (latent, wk, norm_weight, cos_view, sin_view, indices,
               k_cache, k_scale_cache, k_cache, k_scale_cache)
    signature = build_signature(
        OP, tensors,
        specializations={
            "tensor_contract": "supported",
            "norm.weight_dtype": str(norm_weight.dtype).removeprefix("torch."),
            "attribute.norm_eps": str(float(norm_eps)),
        },
        inference=InferenceSpec(
            compute_dtype="bfloat16", accumulator_dtype="float32",
            fusion_ops=("matmul", "rmsnorm", "rope", "dynamic_quantize"),
        ),
    )
    decision = dispatch(signature, current_hardware())
    return run_selected(decision.kernel_id, *tensors, norm_eps=float(norm_eps))


def _meta(latent, wk, norm_weight, rope_cos, rope_sin, indices,
          k_cache, k_scale_cache, norm_eps=1e-6):
    from flashinfer.attention.indexer_prologue import validate_indexer_prologue_k_static

    validate_indexer_prologue_k_static(
        latent, wk, norm_weight, rope_cos, rope_sin, indices, k_cache, k_scale_cache,
        norm_eps, check_device=False)
    return None


def torch_declarations():
    from flashinfer.jit.torch_library import AscendCOp

    return (
        AscendCOp(
            "flashinfer::indexer_prologue_k",
            K_TORCH_SCHEMA,
            _dispatch,
            _meta,
            K_MUTATES_ARGS,
        ),
    )


OPERATOR = OperatorDefinition(
    name="indexer_prologue_k", domain="attention", op=OP,
    public_entry="flashinfer.attention.indexer_prologue:indexer_prologue_k",
    description="Mutation-only K prologue: BF16 projection, RMSNorm, tail RoPE and "
                "per-row INT8/FP16 publication into caller-owned strided caches.",
    torch_declarations=torch_declarations,
)
