"""Public Torch schemas preserving the existing production dispatch guards."""

from functools import lru_cache
from types import SimpleNamespace


def _metadata(kernel_id, tensors):
    from flashinfer.ascendc import load_contract

    _, recipe, _ = load_contract(kernel_id)
    by_name = dict(zip(recipe.tensor_parameter_names, tensors))
    recipe.validate_tensor_metadata(
        shapes={name: tuple(value.shape) for name, value in by_name.items()},
        dtypes={name: str(value.dtype).removeprefix("torch.") for name, value in by_name.items()},
    )
    if any(value.device != tensors[0].device or not value.is_contiguous() for value in tensors):
        raise ValueError("ACLNN tensors must be contiguous and share one device")


def _matmul(a, b, bias):
    from flashinfer.gemm import matmul_bias_silu
    return matmul_bias_silu(a, b, bias, backend="ascendc")


def _matmul_out(a, b, bias, out):
    from flashinfer.gemm import matmul_bias_silu
    matmul_bias_silu(a, b, bias, out=out, backend="ascendc")


def _matmul_meta(a, b, bias):
    import torch
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID

    shape = (a.shape[0] if a.ndim else 0, b.shape[1] if b.ndim >= 2 else 0)
    prospective = SimpleNamespace(shape=shape, dtype=torch.bfloat16, device=a.device,
                                  is_contiguous=lambda: True)
    _metadata(MATMUL_BIAS_SILU_KERNEL_ID, (a, b, bias, prospective))
    return torch.empty(shape, dtype=torch.bfloat16, device=a.device)


def _matmul_out_meta(a, b, bias, out):
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    _metadata(MATMUL_BIAS_SILU_KERNEL_ID, (a, b, bias, out))


def _attention(q, k, v, causal=False, kv_layout="NHD", custom_mask=None):
    from flashinfer.attention.catlass_prefill import single_prefill_with_kv_cache_catlass
    return single_prefill_with_kv_cache_catlass(q, k, v, causal=causal, kv_layout=kv_layout,
                                              custom_mask=custom_mask, backend="ascendc")


def _attention_out(q, k, v, out, causal=False, kv_layout="NHD", custom_mask=None):
    from flashinfer.attention.catlass_prefill import single_prefill_with_kv_cache_catlass
    single_prefill_with_kv_cache_catlass(q, k, v, out=out, causal=causal,
                                       kv_layout=kv_layout, custom_mask=custom_mask, backend="ascendc")


def _attention_out_meta(q, k, v, out, causal=False, kv_layout="NHD", custom_mask=None):
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    if causal is not False or kv_layout != "NHD" or custom_mask is not None:
        raise ValueError("CATLASS prefill requires noncausal, unmasked NHD inputs")
    _metadata(FLASH_ATTENTION_KERNEL_ID, (q, k, v, out))


def _attention_meta(q, k, v, causal=False, kv_layout="NHD", custom_mask=None):
    import torch
    prospective = SimpleNamespace(shape=tuple(q.shape), dtype=torch.float16, device=q.device,
                                  is_contiguous=lambda: True)
    _attention_out_meta(q, k, v, prospective, causal, kv_layout, custom_mask)
    return torch.empty_like(q)


@lru_cache(maxsize=1)
def _declarations():
    from flashinfer.jit.torch_library import AscendCOp
    return (
        AscendCOp("flashinfer::matmul_bias_silu", "(Tensor a, Tensor b, Tensor bias) -> Tensor",
                  _matmul, _matmul_meta),
        AscendCOp("flashinfer::matmul_bias_silu_out",
                  "(Tensor a, Tensor b, Tensor bias, Tensor(a!) out) -> ()",
                  _matmul_out, _matmul_out_meta, ("out",)),
        AscendCOp("flashinfer::single_prefill_with_kv_cache_catlass",
                  '(Tensor q, Tensor k, Tensor v, bool causal=False, str kv_layout="NHD", Tensor? custom_mask=None) -> Tensor',
                  _attention, _attention_meta),
        AscendCOp("flashinfer::single_prefill_with_kv_cache_catlass_out",
                  '(Tensor q, Tensor k, Tensor v, Tensor(a!) out, bool causal=False, str kv_layout="NHD", Tensor? custom_mask=None) -> ()',
                  _attention_out, _attention_out_meta, ("out",)),
    )


def enable():
    from flashinfer.utils import enable_torch_library
    from . import _native

    enable_torch_library(ascendc_ops=_declarations())
    _native.require_native()
