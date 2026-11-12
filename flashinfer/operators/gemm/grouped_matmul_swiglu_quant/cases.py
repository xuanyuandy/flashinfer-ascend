"""Author-owned independent CPU oracle, samples and semantic effect checks."""

def golden(torch, x, weight, scales, token_scale, prefixes, *, bias=None, limit=0.0):
    m, n = x.shape[0], weight.shape[-1]
    y, ys = torch.empty(m, n // 2, dtype=torch.int8), torch.empty(m, dtype=torch.float32)
    start = 0
    for expert, end in enumerate(prefixes):
        if end == start:
            continue
        left, right = x[start:end].int(), weight[expert].int()
        if bias is None:
            values = (left @ right).double() * scales[expert].double()
        else:
            # Source A8W4 MSD stores high and centered low nibbles separately.
            high, low = torch.div(left, 16, rounding_mode="floor"), (left & 15) - 8
            channel = scales[expert] if scales.ndim == 2 else scales[expert, 0]
            if scales.ndim == 2 or scales.shape[1] == 1:
                high_mm = (high @ right).float() * channel.half().float()
                low_mm = (low @ right).float() * channel.half().float()
            else:
                group_size = x.shape[1] // scales.shape[1]
                high_mm = torch.zeros(end - start, n, dtype=torch.float16)
                low_mm = torch.zeros_like(high_mm)
                for group in range(scales.shape[1]):
                    span = slice(group * group_size, (group + 1) * group_size)
                    scale = scales[expert, group].half()
                    high_mm += ((high[:, span] @ right[span]).half() * scale).half()
                    low_mm += ((low[:, span] @ right[span]).half() * scale).half()
            values = ((high_mm.float() * 16 + low_mm.float()) + bias[expert]).double()
        values *= token_scale[start:end].double()[:, None]
        gate, up = values.chunk(2, dim=-1)
        if limit:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        activated = torch.nn.functional.silu(gate) * up
        scale = activated.abs().amax(dim=-1) / 127
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        y[start:end] = torch.round(activated / safe_scale[:, None]).clamp(-127, 127).to(torch.int8)
        ys[start:end] = scale.float()
        start = end
    return y, ys


def sample(torch, torch_npu, *, variant="int8", scale_dtype="float32", layout="NZ", groups=1,
           m=17, k=256, n=256, limit=0.0, exact_scales=False):
    rng = torch.Generator().manual_seed(20261112 + m + k + groups)
    x = torch.randint(-5, 6, (m, k), generator=rng, dtype=torch.int8)
    weight = torch.randint(-5, 6, (3, k, n), generator=rng, dtype=torch.int8)
    scales = (torch.rand((3, n) if groups == 1 else (3, groups, n), generator=rng) * 0.004 + 0.002)
    if exact_scales:
        # Exactly representable FP16 scales isolate clipping from the source
        # MSD oracle's intermediate FP16 scale/product rounding differences.
        scales = torch.round(scales * 1024) / 1024
    token_scale = torch.rand(m, generator=rng) * 0.05 + 0.01
    prefixes = [m // 2, m // 2, m]
    bias = None
    options = dict(swiglu_limit=limit)
    if variant == "int32":
        bias = torch.rand(3, n, generator=rng) * 0.1
        packed = torch.zeros(3, k, n // 8, dtype=torch.int64)
        for nibble in range(8):
            packed |= (weight[..., nibble::8].long() & 15) << (nibble * 4)
        device_weight = packed.int().npu()
        if layout == "NZ":
            # Packing AFTER conversion to NZ produces native INT4 tiles.
            # Casting an ND int32-packed Tensor to NZ tiles int32 lanes and
            # has a different byte order, despite identical Torch metadata.
            nz = torch_npu.npu_format_cast(weight.float().npu(), 29)
            device_weight = torch_npu.npu_quantize(
                nz, torch.ones(1, device=nz.device), None, torch.quint4x2, -1, False)
        device_scales = (scales.contiguous().view(torch.int32).long() & 0xFFFFFFFF).npu()
        options["bias"] = bias.npu()
    else:
        scales = scales.to(getattr(torch, scale_dtype))
        device_weight, device_scales = weight.npu(), scales.npu()
    if layout == "NZ" and variant != "int32":
        device_weight = torch_npu.npu_format_cast(device_weight, 29)
    args = (x.npu(), device_weight, device_scales, token_scale.npu(),
            torch.tensor(prefixes, dtype=torch.int64).npu())
    expected = golden(torch, x, weight, scales, token_scale, prefixes, bias=bias, limit=limit)
    return args, options, expected


CASE_PARAMETERS = (
    dict(scale_dtype="float32", m=17), dict(scale_dtype="float16", m=33, k=512, limit=0.01),
    dict(scale_dtype="bfloat16", m=8, k=7168, n=512),
    dict(variant="int32", layout="ND", m=17), dict(variant="int32", layout="NZ", m=33),
    dict(variant="int32", layout="ND", groups=2, m=17, k=512),
    dict(variant="int32", layout="NZ", groups=2, m=33, k=512, limit=0.01, exact_scales=True),
)
TOLERANCES = ((1, 0.005), (1e-7, 0.01))


def verify_effect(torch, call, args, options, outputs):
    if options.get("swiglu_limit", 0):
        unlimited = call(*args, **dict(options, swiglu_limit=0.0))
        assert not torch.equal(outputs[1], unlimited[1]), "the clipping test must exercise clipping"
