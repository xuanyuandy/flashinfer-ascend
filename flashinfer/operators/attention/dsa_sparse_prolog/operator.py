"""DSA sparse-prolog semantics: Q/KV projection, RMSNorm, scale and RoPE."""

import math

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.dsa_sparse_prolog"
INPUT_NAMES = (
    "token_x", "weight_dq", "weight_uq_qr", "weight_dkv_kr",
    "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos",
)
ATTRIBUTE_NAMES = (
    "rmsnorm_epsilon_cq", "rmsnorm_epsilon_ckv", "mla_scale_q_lora", "mla_scale_kv_lora",
    "qk_nope_head_dim", "is_type_demotion", "apply_rope_fusion",
)
_FLOAT_LIMIT = 3.4028234663852886e38


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _finite(name, value):
    if type(value) not in (float, int) or not math.isfinite(value) or abs(float(value)) > _FLOAT_LIMIT:
        raise ValueError(name + " must be finite and representable as float32")


def _width(weight, hidden):
    shape = tuple(int(d) for d in weight.shape)
    if len(shape) == 2:
        if shape[0] == hidden:
            return shape[1]
        if shape[1] == hidden:
            return shape[0]
        raise ValueError("weight rank-2 shape must place He on one axis")
    if len(shape) == 4:
        if shape[2] != 16 or shape[3] != 16:
            raise ValueError("FRACTAL_NZ weight inner axes must be 16")
        return shape[0] * shape[3]
    raise ValueError("weight rank must be 2 (ND) or 4 (NZ)")


def infer(arguments, layouts):
    token, dq, uq, dkv, gamma_cq, gamma_ckv, rope_sin, rope_cos = (
        arguments[name] for name in INPUT_NAMES
    )
    attrs = {name: arguments[name] for name in ATTRIBUTE_NAMES}
    if len(token.shape) not in (2, 3) or any(int(d) <= 0 for d in token.shape):
        raise ValueError("token_x must be positive [T, He] or [B, S, He]")
    hidden = int(token.shape[-1])
    lead = tuple(int(d) for d in token.shape[:-1])
    compute = _dtype(token)
    if compute not in ("float16", "bfloat16"):
        raise ValueError("token_x must be float16 or bfloat16")
    for name, tensor in (
        ("weight_dq", dq), ("weight_uq_qr", uq), ("weight_dkv_kr", dkv),
        ("rmsnorm_gamma_cq", gamma_cq), ("rmsnorm_gamma_ckv", gamma_ckv),
        ("rope_sin", rope_sin), ("rope_cos", rope_cos),
    ):
        if _dtype(tensor) != compute or any(int(d) <= 0 for d in tensor.shape):
            raise ValueError(name + " must be positive and share token_x dtype")
    if tuple(gamma_cq.shape) != (int(gamma_cq.shape[0]),) or tuple(gamma_ckv.shape) != (int(gamma_ckv.shape[0]),):
        raise ValueError("rmsnorm gammas must be rank-1")
    hcq, hckv = int(gamma_cq.shape[0]), int(gamma_ckv.shape[0])
    if _width(dq, hidden) != hcq:
        raise ValueError("weight_dq output width must equal rmsnorm_gamma_cq")
    if len(rope_sin.shape) != len(token.shape) or len(rope_cos.shape) != len(token.shape):
        raise ValueError("rope_sin and rope_cos must match token_x rank")
    if tuple(int(d) for d in rope_sin.shape[:-1]) != lead or tuple(int(d) for d in rope_cos.shape[:-1]) != lead:
        raise ValueError("rope_sin and rope_cos must match token_x leading axes")
    if int(rope_sin.shape[-1]) != int(rope_cos.shape[-1]):
        raise ValueError("rope_sin and rope_cos must share Dr")
    dr = int(rope_sin.shape[-1])
    head_dim = attrs["qk_nope_head_dim"]
    if type(head_dim) is not int or head_dim < 1 or head_dim > 1024:
        raise ValueError("qk_nope_head_dim must be an int in [1, 1024]")
    if (head_dim + dr) == 0 or _width(uq, hcq) % (head_dim + dr):
        raise ValueError("weight_uq_qr width must be divisible by qk_nope_head_dim + Dr")
    heads = _width(uq, hcq) // (head_dim + dr)
    if _width(dkv, hidden) != hckv + dr:
        raise ValueError("weight_dkv_kr width must equal Hckv + Dr")
    for name in ("rmsnorm_epsilon_cq", "rmsnorm_epsilon_ckv", "mla_scale_q_lora", "mla_scale_kv_lora"):
        _finite(name, attrs[name])
    if type(attrs["is_type_demotion"]) is not bool or type(attrs["apply_rope_fusion"]) is not bool:
        raise ValueError("is_type_demotion and apply_rope_fusion must be bool")
    if layouts is not None:
        for name in INPUT_NAMES:
            allowed = ("ND", "NZ") if name.startswith("weight_") else ("ND",)
            if layouts.get(name, "ND") not in allowed:
                raise ValueError(name + " storage format is outside ND/NZ contract")
    q_nope = lead + (heads, head_dim)
    q_pe = lead + (heads, dr)
    ct_kv = lead + (hckv,)
    k_pe = lead + (1, dr)
    qr = lead + (hcq,)
    rstd = lead + (1,)
    device = token.device
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (
            OutputSpec(q_nope, compute, device), OutputSpec(q_pe, compute, device),
            OutputSpec(ct_kv, compute, device), OutputSpec(k_pe, compute, device),
            OutputSpec(qr, compute, device), OutputSpec(qr, compute, device),
            OutputSpec(rstd, "float32", device), OutputSpec(ct_kv, compute, device),
            OutputSpec(rstd, "float32", device),
        ),
        attrs,
        InferenceSpec(),
    )


def validate_values(arguments):
    """Shape and dtype checks live in infer; values are forwarded to host tiling."""
    return None


OPERATOR = OperatorDefinition(
    name="dsa_sparse_prolog", domain="attention", op=OP,
    schema=(
        "(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_dkv_kr, "
        "Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, "
        "*, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, "
        "float mla_scale_q_lora=0.0, float mla_scale_kv_lora=0.0, int qk_nope_head_dim=128, "
        "bool is_type_demotion=False, bool apply_rope_fusion=False) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
    output_names=("q_nope", "q_pe", "ct_kv", "k_pe", "qr", "q_m", "q_rstd", "ckv_m", "ckv_rstd"),
    infer=infer, validate_values=validate_values,
    description=(
        "DSA sparse-attention prolog. Projects token_x into q_nope, q_pe, ct_kv and k_pe, "
        "and returns the RMSNorm intermediates qr, q_m, q_rstd, ckv_m and ckv_rstd. "
        "Scales <= 0 are derived by the kernel. Not graph capturable."
    ),
)
