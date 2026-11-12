"""Public GroupedMatmulSwigluQuant semantics, without a backend/native contract."""

import math
from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec, QuantSpec

OP = "gemm.grouped_matmul_swiglu_quant"
FUSION_OPS = ("grouped_matmul", "dequantize", "swiglu", "dynamic_quantize")
AUXILIARY_OUTPUTS = ("output_scale",)
FEATURES = (("gemm.mode", "grouped"), ("activation.kind", "silu"))

def quantization(weight_dtype, scale_dtype, layout, group_size=None):
    return (
        (0, QuantSpec("symmetric", "int8", "float32", "int32", granularity="token", axis=(0,), physical_layout="ND")),
        (1, QuantSpec("symmetric", "int4_packed" if weight_dtype == "int32" else "int8",
                      "float32", "int32", scale_dtype="uint64" if weight_dtype == "int32" else scale_dtype,
                      granularity="group" if group_size else "channel", group_size=(group_size,) if group_size else None,
                      axis=(1,) if group_size else (2,), physical_layout=layout,
                      packing_order="low_nibble_first" if weight_dtype == "int32" else None)),
        (6, QuantSpec("symmetric", "int8", "float32", "float32", granularity="token", axis=(0,), physical_layout="ND")),
    )


def infer(arguments, layouts):
    x, weight, scale, token_scale, groups = (arguments[n] for n in
        ("x", "weight", "weight_scale", "x_scale", "group_list"))
    bias, offset, limit = (arguments[n] for n in ("bias", "offset", "swiglu_limit"))
    dtype = str(weight.dtype).removeprefix("torch.")
    if offset is not None:
        raise ValueError("GroupedMatmulSwigluQuant has no supported offset input")
    if dtype not in ("int8", "int32"):
        raise ValueError("weight must be int8 or packed int4 stored in int32")
    packed = dtype == "int32"
    def dtype_name(tensor):
        return str(tensor.dtype).removeprefix("torch.")
    if (dtype_name(x) != "int8" or dtype_name(token_scale) != "float32"
            or dtype_name(groups) != "int64"
            or dtype_name(scale) not in (("int64",) if packed else ("float32", "float16", "bfloat16"))):
        raise ValueError("invalid logical input dtype combination")
    if packed and bias is None:
        raise ValueError("A8W4 requires the float32 weight assistance matrix as bias")
    if not packed and bias is not None:
        raise ValueError("A8W8 does not implement bias")
    if len(x.shape) != 2 or len(weight.shape) != 3 or len(scale.shape) not in (2, 3):
        raise ValueError("x/weight/weight_scale require ranks 2/3/(2 or 3)")
    m, k = x.shape
    e, wk, p = weight.shape
    n = p * (8 if packed else 1)
    count = scale.shape[1] if len(scale.shape) == 3 else 1
    if any(int(v) <= 0 for v in (m, k, e, p)) or n % 2:
        raise ValueError("positive matrix dimensions and an even SwiGLU width are required")
    if tuple(token_scale.shape) != (m,):
        raise ValueError("x_scale dimensions must match M")
    if packed and (tuple(bias.shape) != (e, n) or dtype_name(bias) != "float32"):
        raise ValueError("A8W4 bias assistance matrix requires float32 [E,N]")
    if wk != k or scale.shape[0] != e or scale.shape[-1] != n:
        raise ValueError("weight and weight_scale dimensions do not match x and packed N")
    if packed and (count <= 0 or k % count):
        raise ValueError("A8W4 requires a positive scale group count dividing K")
    if not packed and len(scale.shape) != 2:
        raise ValueError("A8W8 weight_scale is per-channel [E,N]")
    if len(groups.shape) != 1 or not 1 <= groups.shape[0] <= e:
        raise ValueError("group_list must contain between 1 and E expert prefixes")
    layout = layouts["weight"] if layouts is not None else "NZ"
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0:
        raise ValueError("attribute swiglu_limit must be a finite nonnegative number")
    attributes = dict(swiglu_limit=limit)
    inference = InferenceSpec(compute_dtype="int8", accumulator_dtype="int32",
        quantization=quantization(dtype, str(scale.dtype).removeprefix("torch."), layout,
                                  k // count if packed and count > 1 else None),
        fusion_ops=FUSION_OPS, auxiliary_outputs=AUXILIARY_OUTPUTS,
        features=FEATURES + (("activation.clamp", limit > 0),))
    return Invocation((x, weight, scale, token_scale, bias, groups),
                      (OutputSpec((m, n // 2), "int8", x.device), OutputSpec((m,), "float32", x.device)),
                      attributes, inference)


def validate_values(arguments):
    prefixes = arguments["group_list"].cpu().tolist()
    if any(a > b for a, b in zip((0,) + tuple(prefixes), prefixes)) or prefixes[-1] != arguments["x"].shape[0]:
        raise ValueError("group_list must be nondecreasing, nonnegative and end at M")

OPERATOR = OperatorDefinition(
    name="grouped_matmul_swiglu_quant", domain="gemm", op=OP,
    schema="(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale, Tensor group_list, *, Tensor? bias=None, Tensor? offset=None, float swiglu_limit=0.0) -> (Tensor, Tensor)",
    output_names=("output", "output_scale"), infer=infer, validate_values=validate_values,
    description="Grouped integer GEMM, dequantization, SwiGLU and token INT8 quantization. Returns output and output_scale.",
)
