"""DS4.1 MLA prolog: fused projection, norm, quantization and RoPE.

The public contract is one BF16 token block ``x`` of shape ``[T, 5120]`` with
``T`` in ``[1, 4096]`` and a local Q head count ``q_head_num`` in ``[1, 64]``.
``update_kv_cache`` selects the KV write target:

* ``True`` (default): the kernel scatters the computed 512-value row into the
  addressed ``kv_cache`` rows (the fused behavior) and the ``kv`` output is
  allocated but never written; the public entry returns ``(q_output, qr)``.
* ``False``: the kernel never reads ``slot_indices`` and never reads or writes
  ``kv_cache``; the same BF16 row is written to the fresh ``kv [T,512]``
  output and the public entry returns ``(q_output, qr, kv)``. ``kv_cache`` and
  ``slot_indices`` stay required inputs of the native ABI (an optional-input
  OpDef would change the frozen ACLNN signature), but are ignored in this
  mode. The Q side (``q_output``, ``qr`` and the fused Q RoPE) is identical in
  both modes, and the KV row is never split by ``q_head_num``.

The fused chain is:

1. ``x @ w_qa.T`` then RMSNorm with ``gamma_qa``; the same BF16 normalized row
   is the required output ``qr[T, 1280] = q_norm(wq_a(x))``;
2. dynamic per-token INT8 quantization of the normalized Q-LoRA activation;
3. INT8 ``@ w_qb`` with per-token and per-output-channel dequantization;
4. partial RoPE on the last 64 of 512 head channels, producing
   ``q_output[T, q_head_num, 512]``;
5. ``x @ w_kv.T`` then RMSNorm with ``gamma_kv``, partial RoPE, and either an
   in-place scatter of the 512 values into ``kv_cache`` rows addressed by
   ``slot_indices[T, 2]`` (``update_kv_cache=True``) or a plain store of the
   same row to the ``kv`` output (``update_kv_cache=False``).

``rms_norm_eps`` is the FP32 epsilon added to both RMSNorm branches; the
default ``1e-6`` preserves the frozen contract and DS4.1 callers may pass
``1e-20``. It must be finite and strictly positive.

``q_head_num`` is the local Q head count in ``[1, 64]`` (default ``64``). TP
sharding is the caller's responsibility: one device processes only its local
heads and never communicates across cards. ``w_qb`` is
``[q_head_num*16, 80, 16, 32]`` INT8 FRACTAL_NZ, ``w_qb_scale`` is
``[q_head_num*512]``, ``q_output`` is ``[T, q_head_num, 512]`` and ``qr``
stays ``[T, 1280]`` because Q-LoRA is not sharded. The default ``64`` keeps
the previous byte-exact device behavior.

With ``update_kv_cache=True`` the cache rows not addressed by ``slot_indices``
are untouched sentinels. With ``update_kv_cache=False`` the whole cache is
untouched.

IMPORTANT: ``kv_cache`` is an input Tensor that the device kernel updates in
place. The generated functional/out schema path rejects mutation of an input
that is not declared as a mutated argument. This module deliberately keeps the
metadata hook complete, but the final public entry (``public_entry`` plus
``torch_declarations`` with ``mutates_args=("kv_cache",)``, or an explicit
maintainer-approved schema extension) is still unresolved. See
``post_final/flashinfer_submission/gap_analysis.md`` in the evolution output
for the open items; do not treat this declaration as a verified public API.
"""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec, QuantSpec

OP = "attention.mla_prolog"
HIDDEN = 5120
Q_LORA = 1280
KV_LORA = 512
HEAD_DIM = 512
DEFAULT_HEAD_NUM = 64
MAX_HEAD_NUM = 64
MIN_HEAD_NUM = 1
NOPE_DIM = 448
ROPE_DIM = 64
CACHE_PAGES = 22594
CACHE_ROWS = 128
MIN_T = 1
MAX_T = 4096


def q_out_dim(q_head_num=DEFAULT_HEAD_NUM):
    """Q-B output channels (GEMM N) for one token and ``q_head_num`` heads."""
    return int(q_head_num) * HEAD_DIM


def w_qb_nz_shape(q_head_num=DEFAULT_HEAD_NUM):
    """FRACTAL_NZ shape of the INT8 Q-B weight for ``q_head_num`` heads."""
    return (q_out_dim(q_head_num) // 32, Q_LORA // 16, 16, 32)


def _head_num(value):
    """Reject head counts the kernel cannot represent (also rejects bool)."""
    if type(value) is not int or not MIN_HEAD_NUM <= value <= MAX_HEAD_NUM:
        raise ValueError(_HEAD_NUM_ERROR)
    return value


def _update_kv_cache_flag(value):
    """Reject truthy non-bools before the Torch bridge coerces the attribute."""
    if type(value) is not bool:
        raise ValueError(_UPDATE_KV_CACHE_ERROR)
    return value

INPUT_NAMES = (
    "x", "w_qa", "gamma_qa", "w_qb", "w_qb_scale", "w_kv", "gamma_kv",
    "rope_cos", "rope_sin", "slot_indices", "kv_cache",
)
OUTPUT_NAMES = ("q_output", "qr", "kv")
# The native ABI always declares all three outputs (CANN output lists are
# static). The public entry returns (q_output, qr) for update_kv_cache=True and
# (q_output, qr, kv) for False; the kv buffer is allocated but never written in
# the fused mode, so no GM store is added to the fused path.
KV_OUTPUT_INDEX = 2
# rms_norm_eps is a host-side FP32 scalar. The ACLNN attribute carries float64;
# the minimum is the smallest positive float32 normal so zero and negative
# values are rejected at the public boundary.
RMS_NORM_EPS_DEFAULT = 1.0e-6
RMS_NORM_EPS_MINIMUM = 1.1754943508222875e-38
RMS_NORM_EPS_MAXIMUM = 3.4028234663852886e38
UPDATE_KV_CACHE_DEFAULT = True
_HEAD_NUM_ERROR = "q_head_num must be an integer in [1, 64]"
_UPDATE_KV_CACHE_ERROR = "update_kv_cache must be a bool"
FUSION_OPS = (
    "dual_projection", "rms_norm", "dynamic_quantize",
    "quant_matmul", "partial_rope", "kv_cache_scatter",
)
# w_qb is an INT8 weight with one static BF16 scale per logical output channel.
# The other quantized operand is produced inside the fused chain, so it has no
# separate public Tensor binding.
QUANTIZATION = (
    (3, QuantSpec(
        scheme="symmetric", storage_dtype="int8", compute_dtype="int8",
        accumulator_dtype="int32", scale_dtype="bfloat16",
        granularity="channel", axis=(1,), physical_layout="NZ",
    )),
)


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def infer(arguments, layouts):
    """Validate metadata only; never read Tensor values or allocate outputs."""
    x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv, rope_cos, rope_sin, slots, cache = (
        arguments[name] for name in INPUT_NAMES
    )
    rms_norm_eps = arguments.get("rms_norm_eps", RMS_NORM_EPS_DEFAULT)
    if (type(rms_norm_eps) not in (int, float)
            or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0):
        raise ValueError("rms_norm_eps must be finite and > 0")
    rms_norm_eps = float(rms_norm_eps)
    q_head_num = _head_num(arguments.get("q_head_num", DEFAULT_HEAD_NUM))
    update_kv_cache = _update_kv_cache_flag(
        arguments.get("update_kv_cache", UPDATE_KV_CACHE_DEFAULT))
    t = int(x.shape[0]) if len(x.shape) == 2 else -1
    if t < MIN_T or t > MAX_T:
        raise ValueError("x must have shape [T, 5120] with T in [1, 4096]")
    expected = {
        "x": (t, HIDDEN),
        "w_qa": (Q_LORA, HIDDEN),
        "gamma_qa": (Q_LORA,),
        "w_qb": w_qb_nz_shape(q_head_num),
        "w_qb_scale": (q_out_dim(q_head_num),),
        "w_kv": (KV_LORA, HIDDEN),
        "gamma_kv": (KV_LORA,),
        "rope_cos": (t, 1, 1, ROPE_DIM),
        "rope_sin": (t, 1, 1, ROPE_DIM),
        "slot_indices": (t, 2),
        "kv_cache": (CACHE_PAGES, CACHE_ROWS, KV_LORA),
    }
    tensors = {
        "x": x, "w_qa": w_qa, "gamma_qa": gamma_qa, "w_qb": w_qb,
        "w_qb_scale": w_qb_scale, "w_kv": w_kv, "gamma_kv": gamma_kv,
        "rope_cos": rope_cos, "rope_sin": rope_sin, "slot_indices": slots,
        "kv_cache": cache,
    }
    for name, shape in expected.items():
        if _shape(tensors[name]) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {_shape(tensors[name])}")
    dtypes = {
        "x": "bfloat16", "w_qa": "bfloat16", "gamma_qa": "bfloat16",
        "w_qb": "int8", "w_qb_scale": "bfloat16", "w_kv": "bfloat16",
        "gamma_kv": "bfloat16", "rope_cos": "float32", "rope_sin": "float32",
        "slot_indices": "int32", "kv_cache": "bfloat16",
    }
    for name, dtype in dtypes.items():
        if _dtype(tensors[name]) != dtype:
            raise ValueError(f"{name} requires {dtype}, got {_dtype(tensors[name])}")
    if layouts is not None:
        # FRACTAL_NZ is the only non-ND public input; it is the packed INT8
        # weight. kv_cache is logically NCL but physically contiguous ND.
        expected_layouts = {name: "ND" for name in INPUT_NAMES}
        expected_layouts["w_qb"] = "NZ"
        for name, layout in expected_layouts.items():
            if name in layouts and layouts[name] != layout:
                raise ValueError(f"{name} requires {layout} physical layout")
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (
            OutputSpec((t, q_head_num, HEAD_DIM), "bfloat16", x.device),
            OutputSpec((t, Q_LORA), "bfloat16", x.device),
            # Declared in both modes; the fused path leaves it unwritten.
            OutputSpec((t, KV_LORA), "bfloat16", x.device),
        ),
        {
            "rms_norm_eps": rms_norm_eps,
            "q_head_num": q_head_num,
            "update_kv_cache": update_kv_cache,
        },
        InferenceSpec(
            compute_dtype="bfloat16", accumulator_dtype="float32",
            quantization=QUANTIZATION, fusion_ops=FUSION_OPS,
            features=(
                ("normalization.kind", "rms"),
                ("normalization.affine", True),
            ),
        ),
    )


def validate_values(arguments):
    """Reject cache addresses and encoder values the native kernel cannot use.

    Large projection operands are not copied to the host here; the independent
    CPU cases and the standalone validation cover their finiteness. This hook
    checks only the small public Tensor values. With ``update_kv_cache=False``
    the kernel ignores ``slot_indices`` entirely, so its values are not read or
    validated (only the still-required shape/type are checked by ``infer``).
    """
    import torch

    x = arguments["x"]
    t = int(x.shape[0])
    if t < MIN_T or t > MAX_T:
        raise ValueError("T must be in [1, 4096]")
    if _update_kv_cache_flag(
            arguments.get("update_kv_cache", UPDATE_KV_CACHE_DEFAULT)):
        slots = arguments["slot_indices"].detach().cpu()
        if bool((slots[:, 0] < 0).any()) or bool((slots[:, 0] >= CACHE_PAGES).any()):
            raise ValueError("slot page index is out of range")
        if bool((slots[:, 1] < 0).any()) or bool((slots[:, 1] >= CACHE_ROWS).any()):
            raise ValueError("slot row index is out of range")
        if torch.unique(slots, dim=0).shape[0] != t:
            raise ValueError("slot_indices must be unique within one call")
    cos = arguments["rope_cos"].detach().cpu().float().reshape(t, ROPE_DIM)
    sin = arguments["rope_sin"].detach().cpu().float().reshape(t, ROPE_DIM)
    if not torch.equal(cos[:, 0::2], cos[:, 1::2]):
        raise ValueError("rope_cos adjacent even/odd values must be equal")
    if not torch.equal(sin[:, 0::2], sin[:, 1::2]):
        raise ValueError("rope_sin adjacent even/odd values must be equal")
    scale = arguments["w_qb_scale"].detach().cpu().float()
    if bool((scale < 0).any()):
        raise ValueError("w_qb_scale must be non-negative")
    for name in ("gamma_qa", "gamma_kv", "rope_cos", "rope_sin", "w_qb_scale"):
        value = arguments[name].detach().cpu().float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")


def _arguments(x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
               rope_cos, rope_sin, slot_indices, kv_cache, rms_norm_eps,
               q_head_num=DEFAULT_HEAD_NUM,
               update_kv_cache=UPDATE_KV_CACHE_DEFAULT):
    return {
        "x": x, "w_qa": w_qa, "gamma_qa": gamma_qa, "w_qb": w_qb,
        "w_qb_scale": w_qb_scale, "w_kv": w_kv, "gamma_kv": gamma_kv,
        "rope_cos": rope_cos, "rope_sin": rope_sin,
        "slot_indices": slot_indices, "kv_cache": kv_cache,
        "rms_norm_eps": rms_norm_eps, "q_head_num": q_head_num,
        "update_kv_cache": update_kv_cache,
    }


def _check_out(out, qr_out, kv_out, x, q_head_num=DEFAULT_HEAD_NUM):
    import torch
    q_head_num = _head_num(q_head_num)
    t = int(x.shape[0])
    if (tuple(out.shape) != (t, q_head_num, HEAD_DIM)
            or out.dtype != torch.bfloat16 or out.device != x.device):
        raise ValueError(
            "q_output must be bfloat16 [T,%d,512] on the input device"
            % q_head_num)
    if (tuple(qr_out.shape) != (t, Q_LORA)
            or qr_out.dtype != torch.bfloat16 or qr_out.device != x.device):
        raise ValueError(
            "qr must be bfloat16 [T,1280] on the input device")
    # kv is declared in both modes (the fused path leaves it unwritten), so its
    # metadata contract is always checked.
    if (tuple(kv_out.shape) != (t, KV_LORA)
            or kv_out.dtype != torch.bfloat16 or kv_out.device != x.device):
        raise ValueError(
            "kv must be bfloat16 [T,512] on the input device")


def _impl(x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
          rope_cos, rope_sin, slot_indices, kv_cache, out=None, qr_out=None,
          kv_out=None, *, rms_norm_eps=RMS_NORM_EPS_DEFAULT,
          q_head_num=DEFAULT_HEAD_NUM,
          update_kv_cache=UPDATE_KV_CACHE_DEFAULT,
          backend=None, kernel_id=None):
    """Eager dispatch shared by the functional and out torch declarations.

    Always returns the full ``(q_output, qr, kv)`` triple because the Torch
    declaration has three static returns; the public facade truncates the
    fused mode to ``(q_output, qr)``. In fused mode ``kv`` is a fresh unwritten
    buffer and the kernel performs no extra GM store.
    """
    from flashinfer.operators.runtime import allocate_output
    from flashinfer.runtime import build_signature, current_hardware, dispatch
    from flashinfer.runtime.runner import run_selected

    update_kv_cache = _update_kv_cache_flag(update_kv_cache)
    arguments = _arguments(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache, rms_norm_eps, q_head_num,
        update_kv_cache)
    validate_values(arguments)
    invocation = infer(arguments, None)
    if out is None:
        out = allocate_output(invocation.outputs[0])
    if qr_out is None:
        qr_out = allocate_output(invocation.outputs[1])
    if kv_out is None:
        kv_out = allocate_output(invocation.outputs[2])
    _check_out(out, qr_out, kv_out, x, q_head_num)
    tensors = invocation.inputs + (out, qr_out, kv_out)
    request = build_signature(
        OP, tensors, inference=invocation.inference,
        specializations={"tensor_contract": "supported"})
    decision = dispatch(
        request, current_hardware(), backend=backend, kernel_id=kernel_id)
    run_selected(decision.kernel_id, *tensors, **invocation.attributes)
    return out, qr_out, kv_out


def _mla_prolog(x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
                rope_cos, rope_sin, slot_indices, kv_cache,
                *, rms_norm_eps=RMS_NORM_EPS_DEFAULT,
                q_head_num=DEFAULT_HEAD_NUM,
                update_kv_cache=UPDATE_KV_CACHE_DEFAULT,
                backend=None, kernel_id=None):
    return _impl(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache,
        rms_norm_eps=rms_norm_eps, q_head_num=q_head_num,
        update_kv_cache=update_kv_cache,
        backend=backend, kernel_id=kernel_id)


def _mla_prolog_out(x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
                    rope_cos, rope_sin, slot_indices, kv_cache, q_output, qr,
                    kv, *, rms_norm_eps=RMS_NORM_EPS_DEFAULT,
                    q_head_num=DEFAULT_HEAD_NUM,
                    update_kv_cache=UPDATE_KV_CACHE_DEFAULT,
                    backend=None, kernel_id=None):
    _impl(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache, out=q_output, qr_out=qr,
        kv_out=kv, rms_norm_eps=rms_norm_eps, q_head_num=q_head_num,
        update_kv_cache=update_kv_cache,
        backend=backend, kernel_id=kernel_id)


def _mla_prolog_meta(x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv,
                     gamma_kv, rope_cos, rope_sin, slot_indices,
                     kv_cache, *, rms_norm_eps=RMS_NORM_EPS_DEFAULT,
                     q_head_num=DEFAULT_HEAD_NUM,
                     update_kv_cache=UPDATE_KV_CACHE_DEFAULT,
                     backend=None, kernel_id=None):
    import torch
    arguments = _arguments(
        x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
        rope_cos, rope_sin, slot_indices, kv_cache, rms_norm_eps, q_head_num,
        update_kv_cache)
    invocation = infer(arguments, None)
    return tuple(
        torch.empty(specification.shape,
                    dtype=getattr(torch, specification.dtype), device=x.device)
        for specification in invocation.outputs)


def _mla_prolog_out_meta(x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv,
                         gamma_kv, rope_cos, rope_sin, slot_indices,
                         kv_cache, q_output, qr, kv, *,
                         rms_norm_eps=RMS_NORM_EPS_DEFAULT,
                         q_head_num=DEFAULT_HEAD_NUM,
                         update_kv_cache=UPDATE_KV_CACHE_DEFAULT,
                         backend=None, kernel_id=None):
    _check_out(q_output, qr, kv, x, q_head_num)
    return None


def torch_declarations():
    """Rich Torch declarations; kv_cache is an in-place public side effect.

    The three static returns keep the generated ACLNN output list stable. The
    public facade returns the first two in fused mode. ``kv`` is a fresh output
    (never aliases an argument) and is left unwritten when
    ``update_kv_cache=True``.
    """
    from flashinfer.jit.torch_library import AscendCOp

    inputs = (
        "Tensor x, Tensor w_qa, Tensor gamma_qa, Tensor w_qb, "
        "Tensor w_qb_scale, Tensor w_kv, Tensor gamma_kv, Tensor rope_cos, "
        "Tensor rope_sin, Tensor slot_indices, Tensor(a!) kv_cache"
    )
    options = ("*, float rms_norm_eps=1e-6, int q_head_num=64, "
               "bool update_kv_cache=True, str? backend=None, "
               "str? kernel_id=None")
    return (
        AscendCOp(
            "flashinfer::mla_prolog",
            "(%s, %s) -> (Tensor, Tensor, Tensor)" % (inputs, options),
            _mla_prolog,
            _mla_prolog_meta,
            ("kv_cache",),
        ),
        AscendCOp(
            "flashinfer::mla_prolog_out",
            "(%s, Tensor(b!) q_output, Tensor(c!) qr, Tensor(d!) kv, %s) -> ()"
            % (inputs, options),
            _mla_prolog_out,
            _mla_prolog_out_meta,
            ("kv_cache", "q_output", "qr", "kv"),
        ),
    )


OPERATOR = OperatorDefinition(
    name="mla_prolog",
    domain="attention",
    op=OP,
    public_entry="flashinfer.attention.mla_prolog:mla_prolog",
    output_names=OUTPUT_NAMES,
    infer=infer,
    validate_values=validate_values,
    description=(
        "DS4.1 MLA prolog fusion with fixed HIDDEN=5120, Q_LORA=1280, "
        "KV_LORA=512, HEAD_DIM=512, D_NOPE=448 and D_ROPE=64, and a local "
        "Q head count q_head_num in [1,64] (default 64) for TP sharding. "
        "Returns q_output[T,q_head_num,512] and the unsharded "
        "qr[T,1280] = q_norm(wq_a(x)). update_kv_cache (default true) keeps "
        "the fused in-place scatter into the kv_cache rows addressed by "
        "slot_indices and returns (q_output,qr); unaddressed rows are "
        "sentinels. update_kv_cache=false never reads slot_indices and never "
        "reads or writes kv_cache: the same BF16 RMSNorm+partial-RoPE row is "
        "written to the fresh kv[T,512] output and the entry returns "
        "(q_output,qr,kv). q_output/qr and the fused Q RoPE are identical in "
        "both modes and kv is never split by q_head_num. w_qb is "
        "[q_head_num*16,80,16,32] INT8 NZ and w_qb_scale is "
        "[q_head_num*512]. "
        "rms_norm_eps (default 1e-6) parameterizes both RMSNorm branches and "
        "must be finite and > 0. BF16 materialization points and INT8 dynamic "
        "quantization are frozen. T is in [1,4096] and each call is one "
        "launch. kv is a declared output in both modes but the fused path "
        "performs no kv GM store. The public torch declaration lists kv_cache "
        "as a mutated argument. Graph capture is not declared."
    ),
)
