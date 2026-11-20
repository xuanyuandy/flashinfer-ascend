"""Public facade for the A3 DS4.1 MLA prolog fusion.

The fused chain returns ``q_output[T,q_head_num,512]`` and the required
``qr[T,1280] = q_norm(wq_a(x))``. ``update_kv_cache`` selects the KV write
target:

* ``True`` (default): the kernel updates the addressed rows of ``kv_cache`` in
  place and the entry returns ``(q_output, qr)``. ``kv_cache`` is declared as
  a mutated argument on the Torch custom-op bridge; unaddressed rows are
  sentinels and must not change. The declared ``kv`` output is allocated but
  never written, so the fused path gains no GM store.
* ``False``: ``kv_cache`` and ``slot_indices`` are ignored (never read or
  written by the kernel; they remain required native inputs because the CANN
  output/input list is static). The same BF16 RMSNorm+partial-RoPE row is
  written to the fresh ``kv [T,512]`` output and the entry returns
  ``(q_output, qr, kv)``. ``q_output``/``qr`` and the fused Q RoPE are
  identical in both modes and ``kv`` is never split by ``q_head_num``.

``q_head_num`` is the local Q head count in ``[1,64]`` (default ``64``); TP
sharding is the caller's responsibility and the operator never communicates
across cards. ``qr`` is not sharded. ``rms_norm_eps`` parameterizes both
RMSNorm branches (default ``1e-6``; DS4.1 callers may pass ``1e-20``). This
eager entry is not graph capturable.
"""

from __future__ import annotations

from typing import Optional, Union

from ..runtime import Backend

FUSED_OUTPUT_COUNT = 2


def _public_result(outputs, update_kv_cache):
    """Slice the static triple to the fused pair; diagnostics keep all three."""
    if type(update_kv_cache) is not bool:
        raise ValueError("update_kv_cache must be a bool")
    if len(outputs) != 3:
        raise ValueError("the native ABI must produce (q_output, qr, kv)")
    return tuple(outputs) if not update_kv_cache else tuple(
        outputs[:FUSED_OUTPUT_COUNT])


def mla_prolog(
    x,
    w_qa,
    gamma_qa,
    w_qb,
    w_qb_scale,
    w_kv,
    gamma_kv,
    rope_cos,
    rope_sin,
    slot_indices,
    kv_cache,
    out=None,
    qr_out=None,
    kv_out=None,
    *,
    rms_norm_eps: float = 1e-6,
    q_head_num: int = 64,
    update_kv_cache: bool = True,
    backend: Union[str, Backend, None] = None,
    kernel_id: Optional[str] = None,
):
    """Run the DS4.1 MLA prolog chain on Ascend910C.

    ``x`` is BF16 ``[T,5120]`` with T in [1,4096]; ``q_head_num`` is the local
    Q head count in [1,64] (default 64, for TP sharding); ``w_qb`` is INT8
    FRACTAL_NZ ``[q_head_num*16,80,16,32]`` and ``w_qb_scale`` is
    ``[q_head_num*512]``; ``kv_cache`` is BF16 ``[22594,128,512]``.

    ``update_kv_cache=True`` (default) updates the cache rows named by
    ``slot_indices[T,2]`` in place and returns
    ``(q_output[T,q_head_num,512], qr[T,1280])``, both BF16. With
    ``update_kv_cache=False`` the cache and ``slot_indices`` are ignored and
    the entry returns ``(q_output, qr, kv)`` where ``kv[T,512]`` is the BF16
    row the fused path would have written into the cache (RMSNorm then
    interleaved partial RoPE on the last 64 channels). ``qr`` is not sharded
    by ``q_head_num``; ``kv`` is never sharded either.

    ``rms_norm_eps`` must be finite and strictly positive. Passing ``out``
    and/or ``qr_out``/``kv_out`` writes the supplied tensors (other outputs are
    allocated) and returns the pair/triple. ``out`` may be a 2-tuple
    ``(q_output, qr)`` in fused mode or a 3-tuple ``(q_output, qr, kv)``;
    supplying the 3-tuple in fused mode is allowed but ``kv`` stays unwritten.
    The dispatcher rejects unsupported dtypes, shapes, physical layouts, SoC
    and missing AOT artifacts.
    """
    import math
    import torch
    from ..aclnn._ops import ensure_registered

    # Reject non-numeric or non-finite eps before the Torch bridge coerces it.
    if (type(rms_norm_eps) not in (int, float)
            or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0):
        raise ValueError("rms_norm_eps must be finite and > 0")
    # Reject bool/float/out-of-range head counts before the Torch bridge
    # coerces the int attribute.
    if type(q_head_num) is not int or not 1 <= q_head_num <= 64:
        raise ValueError("q_head_num must be an integer in [1, 64]")
    # Reject truthy non-bools (1, "false", ...) before attribute coercion.
    if type(update_kv_cache) is not bool:
        raise ValueError("update_kv_cache must be a bool")
    ensure_registered()
    options = dict(
        backend=backend.value if isinstance(backend, Backend) else backend,
        kernel_id=kernel_id,
        rms_norm_eps=rms_norm_eps,
        q_head_num=q_head_num,
        update_kv_cache=update_kv_cache,
    )
    if out is None and qr_out is None and kv_out is None:
        result = torch.ops.flashinfer.mla_prolog(
            x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
            rope_cos, rope_sin, slot_indices, kv_cache, **options,
        )
        # The Torch declaration has three static returns (the ACLNN output
        # list is static). Fused mode only exposes the legacy pair.
        return _public_result(result, update_kv_cache)
    if isinstance(out, (tuple, list)):
        if len(out) == 2 and qr_out is None and kv_out is None:
            out, qr_out = out
        elif len(out) == 3 and qr_out is None and kv_out is None:
            out, qr_out, kv_out = out
        else:
            raise ValueError(
                "out must be (q_output, qr) or (q_output, qr, kv) alone, "
                "or pass qr_out/kv_out separately")
    t = int(x.shape[0])
    if out is None:
        out = torch.empty(
            (t, q_head_num, 512), dtype=torch.bfloat16, device=x.device)
    if qr_out is None:
        qr_out = torch.empty(
            (t, 1280), dtype=torch.bfloat16, device=x.device)
    if kv_out is None:
        # Allocated in fused mode too because the native output list is
        # static; the kernel leaves it unwritten.
        kv_out = torch.empty(
            (t, 512), dtype=torch.bfloat16, device=x.device)
    torch.ops.flashinfer.mla_prolog_out(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache, out, qr_out, kv_out,
        **options,
    )
    return _public_result((out, qr_out, kv_out), update_kv_cache)


__all__ = ["mla_prolog"]
