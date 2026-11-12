"""Backend-independent declaration for ascendc.single_prefill_catlass."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(
    name="single_prefill_catlass",
    domain="attention",
    op="ascendc.single_prefill_catlass",
    public_entry="flashinfer.attention.catlass_prefill:single_prefill_with_kv_cache_catlass",
    tensor_shapes=(("S", "H", "D"), ("T", "H", "D"), ("T", "H", "D"), ("S", "H", "D")),
)


from types import SimpleNamespace


def _attention(
    q,
    k,
    v,
    causal=False,
    kv_layout="NHD",
    custom_mask=None,
    *,
    backend="auto",
    kernel_id=None
):
    from flashinfer.attention.catlass_prefill import _single_prefill_impl

    return _single_prefill_impl(
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        custom_mask=custom_mask,
        backend=backend,
        kernel_id=kernel_id,
    )


def _attention_out(
    q,
    k,
    v,
    out,
    causal=False,
    kv_layout="NHD",
    custom_mask=None,
    *,
    backend="auto",
    kernel_id=None
):
    from flashinfer.attention.catlass_prefill import _single_prefill_impl

    _single_prefill_impl(
        q,
        k,
        v,
        out=out,
        causal=causal,
        kv_layout=kv_layout,
        custom_mask=custom_mask,
        backend=backend,
        kernel_id=kernel_id,
    )


def _attention_out_meta(
    q,
    k,
    v,
    out,
    causal=False,
    kv_layout="NHD",
    custom_mask=None,
    *,
    backend="auto",
    kernel_id=None
):
    if causal is not False or kv_layout != "NHD" or custom_mask is not None:
        raise ValueError("CATLASS prefill requires noncausal, unmasked NHD inputs")
    if (
        len(q.shape) != 3
        or len(k.shape) != 3
        or tuple(k.shape) != tuple(v.shape)
        or tuple(out.shape) != tuple(q.shape)
        or q.shape[1:] != k.shape[1:]
        or any(t.dtype != q.dtype or t.device != q.device for t in (k, v, out))
    ):
        raise ValueError("invalid dense prefill Tensor metadata")


def _attention_meta(
    q,
    k,
    v,
    causal=False,
    kv_layout="NHD",
    custom_mask=None,
    *,
    backend="auto",
    kernel_id=None
):
    import torch

    prospective = SimpleNamespace(
        shape=tuple(q.shape), dtype=q.dtype, device=q.device, is_contiguous=lambda: True
    )
    _attention_out_meta(q, k, v, prospective, causal, kv_layout, custom_mask)
    return torch.empty_like(q)


def torch_declarations():
    from flashinfer.jit.torch_library import AscendCOp

    return (
        AscendCOp(
            "flashinfer::single_prefill_with_kv_cache_catlass",
            '(Tensor q, Tensor k, Tensor v, bool causal=False, str kv_layout="NHD", Tensor? custom_mask=None, *, str? backend="auto", str? kernel_id=None) -> Tensor',
            _attention,
            _attention_meta,
        ),
        AscendCOp(
            "flashinfer::single_prefill_with_kv_cache_catlass_out",
            '(Tensor q, Tensor k, Tensor v, Tensor(a!) out, bool causal=False, str kv_layout="NHD", Tensor? custom_mask=None, *, str? backend="auto", str? kernel_id=None) -> ()',
            _attention_out,
            _attention_out_meta,
            ("out",),
        ),
    )


def metadata_reason(q, k, v, out):
    if (
        len(q.shape) != 3
        or len(k.shape) != 3
        or tuple(k.shape) != tuple(v.shape)
        or tuple(out.shape) != tuple(q.shape)
        or q.shape[1:] != k.shape[1:]
        or any(t.dtype != q.dtype or t.device != q.device for t in (k, v, out))
    ):
        return "inconsistent dense prefill Tensor metadata"
    return "supported"
