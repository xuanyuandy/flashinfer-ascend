"""Fused four-way residual mix, mix-coefficient update and RMSNorm projection."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

MF = 24  # projected coefficient rows: 4(pre) + 4(post) + 16(comb)
STREAMS = 4


def infer(arguments, layouts):
    x = arguments["x"]
    residual = arguments["residual"]
    previous_mix = arguments["previous_mix"]
    post_mix = arguments["post_mix"]
    comb_res_mix = arguments["comb_res_mix"]
    fn = arguments["fn"]
    mix_scales = arguments["mix_scales"]
    mix_bases = arguments["mix_bases"]
    rmsnorm_weight = arguments["rmsnorm_weight"]
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError("x requires positive [T, H] dimensions")
    t, h = x.shape
    for tensor, name, shape in (
        (residual, "residual", (t, STREAMS, h)),
        (previous_mix, "previous_mix", (t, STREAMS)),
        (post_mix, "post_mix", (t, STREAMS, 1)),
        (comb_res_mix, "comb_res_mix", (t, STREAMS, STREAMS)),
        (fn, "fn", (MF, STREAMS * h)),
        (mix_scales, "mix_scales", (3,)),
        (mix_bases, "mix_bases", (MF,)),
        (rmsnorm_weight, "rmsnorm_weight", (h,)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError("%s requires shape %s" % (name, shape))
    for tensor, name, dtype in (
        (x, "x", "bfloat16"),
        (residual, "residual", "bfloat16"),
        (previous_mix, "previous_mix", "float32"),
        (post_mix, "post_mix", "float32"),
        (comb_res_mix, "comb_res_mix", "float32"),
        (fn, "fn", "float32"),
        (mix_scales, "mix_scales", "float32"),
        (mix_bases, "mix_bases", "float32"),
        (rmsnorm_weight, "rmsnorm_weight", "bfloat16"),
    ):
        if str(tensor.dtype).removeprefix("torch.") != dtype:
            raise ValueError("%s requires %s dtype" % (name, dtype))
    outputs = (
        OutputSpec((t, STREAMS, h), "bfloat16", x.device),
        OutputSpec((t, STREAMS, 1), "float32", x.device),
        OutputSpec((t, STREAMS, STREAMS), "float32", x.device),
        OutputSpec((t, h), "bfloat16", x.device),
        OutputSpec((t, STREAMS), "float32", x.device),
    )
    return Invocation(
        (x, residual, previous_mix, post_mix, comb_res_mix, fn, mix_scales,
         mix_bases, rmsnorm_weight),
        outputs,
        {},
        InferenceSpec(compute_dtype="float32", accumulator_dtype="float32"),
    )


OPERATOR = OperatorDefinition(
    name="mega_mhc",
    domain="mhc",
    op="mhc.mega_mhc",
    schema=(
        "(Tensor x, Tensor residual, Tensor previous_mix, Tensor post_mix, "
        "Tensor comb_res_mix, Tensor fn, Tensor mix_scales, Tensor mix_bases, "
        "Tensor rmsnorm_weight) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
    output_names=("new_residual", "new_post_mix", "new_comb_res_mix", "y",
                  "new_previous_mix"),
    infer=infer,
    tensor_shapes=(
        ("T", "H"), ("T", "R", "H"), ("T", "R"), ("T", "R", "ONE"),
        ("T", "R", "R"), ("MF", "C"), ("THREE",), ("MF",), ("H",),
        ("T", "R", "H"), ("T", "R", "ONE"), ("T", "R", "R"), ("T", "H"),
        ("T", "R"),
    ),
    description=(
        "Mix four residual streams, update the pre/post/combination mix "
        "coefficients (sigmoid + 4x4 Sinkhorn) and normalize the result with a "
        "24x4H FP32 projection."
    ),
)
