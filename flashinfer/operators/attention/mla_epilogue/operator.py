"""Backend-independent semantics for MLA decode epilogue fusion."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "attention.mla_epilogue"
INPUT_NAMES = ("x", "cos", "sin", "w1", "w2")


def infer(arguments, layouts):
    x, cos, sin, w1, w2 = (arguments[name] for name in INPUT_NAMES)
    if len(x.shape) != 4 or x.shape[1:] != (1, 64, 512):
        raise ValueError("x must have shape [S,1,64,512]")
    s = x.shape[0]
    if not 1 <= s <= 131072:
        raise ValueError("S must be in [1,131072]")
    if tuple(cos.shape) != (s, 1, 1, 64) or tuple(sin.shape) != (s, 1, 1, 64):
        raise ValueError("cos/sin must have shape [S,1,1,64] and match x.shape[0]")
    if tuple(w1.shape) != (8, 4096, 1024):
        raise ValueError("w1 must have shape [8,4096,1024]")
    if tuple(w2.shape) != (5120, 8192):
        raise ValueError("w2 must have shape [5120,8192]")
    if any(str(t.dtype).removeprefix("torch.") != "bfloat16" for t in (x, w1, w2)):
        raise ValueError("x/w1/w2 must be bfloat16")
    if any(str(t.dtype).removeprefix("torch.") != "float32" for t in (cos, sin)):
        raise ValueError("cos/sin must be float32")
    if layouts is not None and any(value != "ND" for value in layouts.values()):
        raise ValueError("MlaEpilogue requires contiguous ND tensors")
    return Invocation(
        tuple(arguments[name] for name in INPUT_NAMES),
        (OutputSpec((s, 5120), "bfloat16", x.device),),
        {},
        InferenceSpec(compute_dtype="bfloat16", accumulator_dtype="float32", attention_variant="mla"),
    )


OPERATOR = OperatorDefinition(
    name="mla_epilogue",
    domain="attention",
    op=OP,
    schema="(Tensor x, Tensor cos, Tensor sin, Tensor w1, Tensor w2) -> Tensor",
    output_names=("y",),
    infer=infer,
    description=(
        "MLA decode epilogue: interleave RoPE over the last 64 dimensions of each head, "
        "followed by eight batched GEMMs and one transposed-B GEMM. Inputs are contiguous "
        "ND tensors with S in [1,131072]. Not graph capturable."
    ),
)
