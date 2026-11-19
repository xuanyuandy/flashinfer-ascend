"""Author-owned independent CPU oracle, samples and semantic effect checks."""

# The device kernel emits one [M, N] tile per group and the runtime clears the
# output before launch, so an empty group must read back as zeros.


def golden(torch, x, weight, prefixes, *, trans_a=False, out_dtype=None):
    """Per-group reference: group g multiplies K rows [start, end) of both operands."""
    total_k, n = int(weight.shape[0]), int(weight.shape[1])
    m = int(x.shape[1]) if trans_a else int(x.shape[0])
    dtype = out_dtype or x.dtype
    y = torch.zeros((len(prefixes), m, n), dtype=dtype)
    start = 0
    for group, end in enumerate(prefixes):
        end = int(end)
        if end > start:
            left = x[start:end].double().T if trans_a else x[:, start:end].double()
            y[group] = (left @ weight[start:end].double()).to(dtype)
        start = end
    if start != total_k:
        raise ValueError("prefixes must end at the K extent")
    return y


def sample(torch, torch_npu, *, m=64, n=96, groups=(32, 0, 48, 16), dtype="float16",
           trans_a=False, fp32_output=False):
    """Build one case; `groups` lists per-group K lengths so empty groups stay expressible."""
    groups = tuple(int(value) for value in groups)
    if any(value < 0 for value in groups):
        raise ValueError("group lengths must be non-negative")
    total_k = sum(groups)
    prefixes = []
    running = 0
    for value in groups:
        running += value
        prefixes.append(running)

    rng = torch.Generator().manual_seed(20261118 + m + n + total_k + len(groups))
    torch_dtype = getattr(torch, dtype)
    weight = torch.randn((total_k, n), generator=rng).to(torch_dtype)
    x = torch.randn((total_k, m) if trans_a else (m, total_k), generator=rng).to(torch_dtype)
    group_list = torch.tensor(prefixes, dtype=torch.int64)

    args = (x.npu(), weight.npu(), group_list.npu())
    options = dict(trans_a=trans_a, fp32_output=fp32_output)
    expected = golden(torch, x, weight, prefixes, trans_a=trans_a,
                      out_dtype=torch.float32 if fp32_output else torch_dtype)
    return args, options, expected


# Covers both transposed states, both input dtypes, both output dtypes, empty
# groups and a single group. Every group length is a multiple of 16, matching the
# range the upstream operator was validated on.
CASE_PARAMETERS = (
    dict(m=64, n=96, groups=(32, 0, 48, 16)),
    dict(m=64, n=96, groups=(32, 0, 48, 16), trans_a=True),
    dict(m=64, n=96, groups=(32, 0, 48, 16), dtype="bfloat16"),
    dict(m=64, n=96, groups=(32, 0, 48, 16), dtype="bfloat16", trans_a=True),
    dict(m=64, n=96, groups=(32, 0, 48, 16), fp32_output=True),
    dict(m=64, n=96, groups=(32, 0, 48, 16), fp32_output=True, trans_a=True),
    dict(m=128, n=256, groups=(128, 128, 192, 192), dtype="bfloat16", fp32_output=True),
    dict(m=32, n=64, groups=(16, 16, 16, 16), dtype="bfloat16", trans_a=True),
    dict(m=64, n=96, groups=(192,)),
)

# float32 output reuses the same accumulation and only widens the store, so the
# looser bound is driven by the bfloat16 cases.
TOLERANCES = ((1e-2, 2e-2),)

# Tighter bounds for the cases whose inputs and outputs are float16.
STRICT_TOLERANCES = ((2e-3, 2e-3),)


def tolerances(configuration):
    dtype = configuration.get("dtype", "float16")
    return STRICT_TOLERANCES if dtype == "float16" else TOLERANCES


def verify_effect(torch, call, args, options, outputs):
    """An empty group must still produce a zero tile, not stale memory."""
    prefixes = [int(v) for v in args[2].cpu().tolist()]
    start = 0
    for group, end in enumerate(prefixes):
        if end == start:
            assert torch.count_nonzero(outputs[0][group]) == 0, "empty group must be zero-filled"
        start = end
