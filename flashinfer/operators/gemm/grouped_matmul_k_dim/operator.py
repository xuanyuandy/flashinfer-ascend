"""Backend-independent declaration for gemm.grouped_matmul_k_dim."""

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import CapabilityCase, InferenceSpec, ShapeRange

OP = "gemm.grouped_matmul_k_dim"

# Ascend C matmul reads ND operands through 16-bit strides.
MAX_STRIDE = 65535
_INPUT_DTYPES = ("float16", "bfloat16")


def capability_cases():
    """One coherent AND clause per (compute dtype, output dtype) pair.

    Shared by the native and reference backends: an explicit InferenceSpec is only
    admissible against declared capability cases, and both backends compute the
    same thing.
    """
    cases = []
    for compute in _INPUT_DTYPES:
        for output in (compute, "float32"):
            cases.append(
                CapabilityCase(
                    name="%s_to_%s" % (compute, output),
                    dtype_signatures=((compute, compute, "int64", output),),
                    compute_dtypes=(compute,),
                    accumulator_dtypes=("float32",),
                    # x keeps its row stride within the 16-bit matmul stride limit in
                    # both transposed states; N is shared by weight and output.
                    shape_ranges=(
                        ShapeRange("XB", 0, 1, 1, MAX_STRIDE),
                        ShapeRange("N", 1, 1, 1, MAX_STRIDE),
                    ),
                    specializations=(("tensor_contract", ("supported",)),),
                )
            )
    return tuple(cases)


def _dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def infer(arguments, layouts):
    x, weight, group_list = (arguments[name] for name in ("x", "weight", "group_list"))
    trans_a = arguments["trans_a"]
    fp32_output = arguments["fp32_output"]

    if len(x.shape) != 2 or len(weight.shape) != 2 or len(group_list.shape) != 1:
        raise ValueError("x and weight require rank 2, group_list requires rank 1")

    dtype = _dtype_name(x)
    if dtype not in _INPUT_DTYPES:
        raise ValueError("x must be float16 or bfloat16")
    if _dtype_name(weight) != dtype:
        raise ValueError("weight dtype must match x")
    if _dtype_name(group_list) != "int64":
        raise ValueError("group_list must be int64 cumulative prefixes over K")

    m = int(x.shape[1] if trans_a else x.shape[0])
    k_a = int(x.shape[0] if trans_a else x.shape[1])
    k_b, n = (int(v) for v in weight.shape)
    groups = int(group_list.shape[0])
    if any(value <= 0 for value in (m, k_a, k_b, n, groups)):
        raise ValueError("M, K, N and the group count must be positive")
    if k_a != k_b:
        # The grouping splits the shared K axis, so x and weight must agree on it.
        raise ValueError("x and weight disagree on the K extent")
    if int(x.shape[1]) > MAX_STRIDE or n > MAX_STRIDE:
        raise ValueError("x row stride and N must not exceed %d" % MAX_STRIDE)

    out_dtype = "float32" if fp32_output else dtype
    attributes = dict(trans_a=bool(trans_a), fp32_output=bool(fp32_output))
    inference = InferenceSpec(compute_dtype=dtype, accumulator_dtype="float32")
    return Invocation(
        (x, weight, group_list),
        (OutputSpec((groups, m, n), out_dtype, x.device),),
        attributes,
        inference,
    )


def validate_values(arguments):
    """Reject a malformed prefix array before any device work is queued."""
    prefixes = [int(value) for value in arguments["group_list"].cpu().tolist()]
    total_k = int(arguments["weight"].shape[0])
    if not prefixes:
        raise ValueError("group_list must contain at least one group")
    start = 0
    for value in prefixes:
        if value < start:
            raise ValueError("group_list must be non-decreasing cumulative prefixes")
        start = value
    if prefixes[-1] != total_k:
        raise ValueError("group_list must end at the K extent of weight (%d)" % total_k)


OPERATOR = OperatorDefinition(
    name="grouped_matmul_k_dim",
    domain="gemm",
    op=OP,
    schema="(Tensor x, Tensor weight, Tensor group_list, *, bool trans_a=False, bool fp32_output=False) -> Tensor",
    output_names=("y",),
    infer=infer,
    validate_values=validate_values,
    description=(
        "Grouped matmul over the shared K axis: group_list holds cumulative K prefixes, "
        "and each group emits its own [M, N] tile, so y is [groups, M, N]."
    ),
)
