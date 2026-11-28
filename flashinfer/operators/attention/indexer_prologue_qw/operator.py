"""One public QW contract for dense BF16 and caller-prequantized INT8 NZ weights.

``prequantized=False`` keeps the original dense contract: the launch performs
weight DynamicQuant from a BF16 ND weight. ``prequantized=True`` consumes a
caller-owned INT8 FRACTAL_NZ weight plus its FP32 scale and skips weight
preprocessing. Both forms share the same activation quantization, Q/W
projection, RoPE, query quantization and output tuple.
"""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.indexer_prologue_qw"
N_HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64
DIM_X = 5120
DIM_QR = 1280
DIM_Q = N_HEADS * HEAD_DIM
DEFAULT_SOFTMAX_SCALE = 128 ** -0.5


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _validate_common(x, qr, ww, rope_sin, rope_cos, tokens):
    expected = (
        ("x", x, "bfloat16", (tokens, DIM_X)),
        ("qr", qr, "bfloat16", (tokens, DIM_QR)),
        ("ww", ww, "bfloat16", (N_HEADS, DIM_X)),
        ("rope_sin", rope_sin, "float32", (tokens, ROPE_DIM)),
        ("rope_cos", rope_cos, "float32", (tokens, ROPE_DIM)),
    )
    for name, tensor, dtype, shape in expected:
        if _dtype(tensor) != dtype:
            raise ValueError("%s must be %s" % (name, dtype))
        if tuple(tensor.shape) != shape:
            raise ValueError("%s must have shape %s; got %s" % (name, shape, tuple(tensor.shape)))


def infer(arguments, layouts):
    x, qr, wqb, ww, rope_sin, rope_cos = (
        arguments[name] for name in ("x", "qr", "wqb", "ww", "rope_sin", "rope_cos")
    )
    wqb_scale = arguments.get("wqb_scale")
    prequantized = bool(arguments.get("prequantized", False))
    softmax_scale = arguments.get("softmax_scale", DEFAULT_SOFTMAX_SCALE)
    tokens = int(x.shape[0]) if len(x.shape) else 0
    if tokens < 1:
        raise ValueError("IndexerPrologueQW requires T >= 1")
    _validate_common(x, qr, ww, rope_sin, rope_cos, tokens)

    if prequantized:
        if wqb_scale is None:
            raise ValueError("prequantized QW requires wqb_scale")
        if _dtype(wqb) != "int8" or tuple(wqb.shape) != (DIM_QR, DIM_Q):
            raise ValueError("prequantized wqb must be int8 with shape (%d, %d)"
                             % (DIM_QR, DIM_Q))
        if _dtype(wqb_scale) != "float32" or tuple(wqb_scale.shape) != (DIM_Q,):
            raise ValueError("prequantized wqb_scale must be float32 with shape (%d,)" % DIM_Q)
        if layouts is not None:
            import torch
            npu = torch._C._get_privateuse1_backend_name()
            if wqb.device.type == npu and layouts.get("wqb") != "NZ":
                raise ValueError("prequantized wqb must carry physical FRACTAL_NZ")
            if layouts.get("wqb_scale") != "ND":
                raise ValueError("prequantized wqb_scale requires physical ND")
            if any(layouts.get(name) != "ND" for name in
                   ("x", "qr", "ww", "rope_sin", "rope_cos")):
                raise ValueError("prequantized QW activations require physical ND")
        inputs = (x, qr, wqb, wqb_scale, ww, rope_sin, rope_cos)
        inference = InferenceSpec(
            compute_dtype="int8", accumulator_dtype="int32",
            fusion_ops=("quantized_matmul", "rope", "dynamic_quantize"),
            auxiliary_outputs=("q_scale", "w"),
        )
        specializations = (("quant.weight", "caller_prequantized_nz"),)
    else:
        if wqb_scale is not None:
            raise ValueError("wqb_scale is only valid when prequantized=True")
        if _dtype(wqb) != "bfloat16" or tuple(wqb.shape) != (DIM_Q, DIM_QR):
            raise ValueError("dense wqb must be bfloat16 with shape (%d, %d)"
                             % (DIM_Q, DIM_QR))
        if layouts is not None and any(layouts.get(name) != "ND" for name in
                                       ("x", "qr", "wqb", "ww", "rope_sin", "rope_cos")):
            raise ValueError("dense QW weights and activations require physical ND")
        inputs = (x, qr, wqb, ww, rope_sin, rope_cos)
        inference = InferenceSpec(
            compute_dtype="int8", accumulator_dtype="int32",
            fusion_ops=("dynamic_quantize", "quantized_matmul", "rope", "dynamic_quantize"),
            auxiliary_outputs=("q_scale", "w"),
        )
        specializations = (("quant.activation", "per_token_int8"),
                           ("quant.weight", "per_channel_int8"))

    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise ValueError("softmax_scale must be a finite float")
    if not math.isfinite(float(softmax_scale)):
        raise ValueError("softmax_scale must be a finite float")
    return Invocation(
        inputs,
        (
            OutputSpec((tokens, N_HEADS, HEAD_DIM), "int8", x.device),
            OutputSpec((tokens, N_HEADS), "float16", x.device),
            OutputSpec((tokens, N_HEADS), "float16", x.device),
        ),
        {"softmax_scale": float(softmax_scale)},
        inference,
        specializations=specializations,
    )


def validate_values(arguments):
    softmax_scale = arguments.get("softmax_scale", DEFAULT_SOFTMAX_SCALE)
    if not isinstance(softmax_scale, bool) and math.isfinite(float(softmax_scale)):
        return None
    raise ValueError("softmax_scale must be a finite float")


OPERATOR = OperatorDefinition(
    name="indexer_prologue_qw", domain="attention", op=OP,
    schema="(Tensor x, Tensor qr, Tensor wqb, Tensor ww, Tensor rope_sin, Tensor rope_cos, "
           "float softmax_scale=0.08838834764831845, *, Tensor? wqb_scale=None, "
           "bool prequantized=False) -> (Tensor, Tensor, Tensor)",
    output_names=("q", "q_scale", "w"), infer=infer, validate_values=validate_values,
    description="Mixed W8A8 Q projection, query RoPE/quantization and W projection. "
                "Dense BF16 weights are dynamically quantized inside the launch; "
                "prequantized=True consumes caller-owned INT8 NZ weights and FP32 scale.",
)
