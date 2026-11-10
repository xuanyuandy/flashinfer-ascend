"""Public Torch schemas preserving the existing production dispatch guards."""

from functools import lru_cache
from types import SimpleNamespace

_registered = False


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


def _matmul(a, b, bias, *, backend="auto", kernel_id=None):
    from flashinfer.gemm import _matmul_bias_silu_impl
    return _matmul_bias_silu_impl(a, b, bias, backend=backend, kernel_id=kernel_id)


def _plain_matmul(a, b, *, backend=None, kernel_id=None):
    from flashinfer.gemm import _matmul_impl
    return _matmul_impl(a, b, backend=backend, kernel_id=kernel_id)


def _plain_matmul_out(a, b, out, *, backend=None, kernel_id=None):
    from flashinfer.gemm import _matmul_impl
    _matmul_impl(a, b, out, backend=backend, kernel_id=kernel_id)


def _plain_matmul_out_meta(a, b, out, *, backend=None, kernel_id=None):
    from flashinfer._matmul import AOT_KERNEL_ID
    _metadata(AOT_KERNEL_ID, (a, b, out))


def _plain_matmul_meta(a, b, *, backend=None, kernel_id=None):
    import torch
    from flashinfer._matmul import output_metadata

    prospective = output_metadata(a, b)
    _plain_matmul_out_meta(a, b, prospective)
    return torch.empty(prospective.shape, dtype=a.dtype, device=a.device)


def _matmul_out(a, b, bias, out, *, backend="auto", kernel_id=None):
    from flashinfer.gemm import _matmul_bias_silu_impl
    _matmul_bias_silu_impl(a, b, bias, out=out, backend=backend, kernel_id=kernel_id)


def _matmul_meta(a, b, bias, *, backend="auto", kernel_id=None):
    import torch
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID

    shape = (a.shape[0] if a.ndim else 0, b.shape[1] if b.ndim >= 2 else 0)
    prospective = SimpleNamespace(shape=shape, dtype=torch.bfloat16, device=a.device,
                                  is_contiguous=lambda: True)
    _metadata(MATMUL_BIAS_SILU_KERNEL_ID, (a, b, bias, prospective))
    return torch.empty(shape, dtype=torch.bfloat16, device=a.device)


def _matmul_out_meta(a, b, bias, out, *, backend="auto", kernel_id=None):
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID
    _metadata(MATMUL_BIAS_SILU_KERNEL_ID, (a, b, bias, out))


def _attention(q, k, v, causal=False, kv_layout="NHD", custom_mask=None, *, backend="auto", kernel_id=None):
    from flashinfer.attention.catlass_prefill import _single_prefill_impl
    return _single_prefill_impl(q, k, v, causal=causal, kv_layout=kv_layout,
                                custom_mask=custom_mask, backend=backend, kernel_id=kernel_id)


def _attention_out(q, k, v, out, causal=False, kv_layout="NHD", custom_mask=None, *, backend="auto", kernel_id=None):
    from flashinfer.attention.catlass_prefill import _single_prefill_impl
    _single_prefill_impl(q, k, v, out=out, causal=causal, kv_layout=kv_layout,
                         custom_mask=custom_mask, backend=backend, kernel_id=kernel_id)


def _attention_out_meta(q, k, v, out, causal=False, kv_layout="NHD", custom_mask=None, *, backend="auto", kernel_id=None):
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
    if causal is not False or kv_layout != "NHD" or custom_mask is not None:
        raise ValueError("CATLASS prefill requires noncausal, unmasked NHD inputs")
    _metadata(FLASH_ATTENTION_KERNEL_ID, (q, k, v, out))


def _attention_meta(q, k, v, causal=False, kv_layout="NHD", custom_mask=None, *, backend="auto", kernel_id=None):
    import torch
    prospective = SimpleNamespace(shape=tuple(q.shape), dtype=q.dtype, device=q.device,
                                  is_contiguous=lambda: True)
    _attention_out_meta(q, k, v, prospective, causal, kv_layout, custom_mask)
    return torch.empty_like(q)


@lru_cache(maxsize=1)
def _declarations():
    from flashinfer.jit.torch_library import AscendCOp
    return (
        AscendCOp("flashinfer::matmul", '(Tensor a, Tensor b, *, str? backend=None, str? kernel_id=None) -> Tensor',
                  _plain_matmul, _plain_matmul_meta),
        AscendCOp("flashinfer::matmul_out",
                  '(Tensor a, Tensor b, Tensor(a!) out, *, str? backend=None, str? kernel_id=None) -> ()',
                  _plain_matmul_out, _plain_matmul_out_meta, ("out",)),
        AscendCOp("flashinfer::matmul_bias_silu", '(Tensor a, Tensor b, Tensor bias, *, str? backend="auto", str? kernel_id=None) -> Tensor',
                  _matmul, _matmul_meta),
        AscendCOp("flashinfer::matmul_bias_silu_out",
                  '(Tensor a, Tensor b, Tensor bias, Tensor(a!) out, *, str? backend="auto", str? kernel_id=None) -> ()',
                  _matmul_out, _matmul_out_meta, ("out",)),
        AscendCOp("flashinfer::single_prefill_with_kv_cache_catlass",
                  '(Tensor q, Tensor k, Tensor v, bool causal=False, str kv_layout="NHD", Tensor? custom_mask=None, *, str? backend="auto", str? kernel_id=None) -> Tensor',
                  _attention, _attention_meta),
        AscendCOp("flashinfer::single_prefill_with_kv_cache_catlass_out",
                  '(Tensor q, Tensor k, Tensor v, Tensor(a!) out, bool causal=False, str kv_layout="NHD", Tensor? custom_mask=None, *, str? backend="auto", str? kernel_id=None) -> ()',
                  _attention_out, _attention_out_meta, ("out",)),
    )


def enable():
    global _registered
    if _registered:
        return
    import torch
    from flashinfer.utils import _registration_lock
    from flashinfer._torch_library import _register_ascendc_ops, _validate_ascendc_ops

    if not (hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_fake")):
        raise RuntimeError("enable_aclnn requires torch >= 2.4")
    with _registration_lock:
        if _registered:
            return
        declarations = _declarations()
        _validate_ascendc_ops(declarations)
        _register_ascendc_ops(declarations)
        _registered = True
