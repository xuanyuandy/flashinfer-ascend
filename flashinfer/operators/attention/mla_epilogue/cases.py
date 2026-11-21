"""Independent CPU oracle and deterministic Atlas A3 acceptance inputs.

Sparse but nonzero w1/w2 keep the full native GEMM shapes while making an
independent FP64 reference inexpensive. Every output element still depends on
RoPE and both GEMM stages.
"""


def golden(torch, x, cos, sin, w1, w2):
    s = x.shape[0]
    rope = x.float().clone()
    rotary = rope[..., 448:512].clone()
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    c, sn = cos.float().reshape(s, 1, 1, 64), sin.float().reshape(s, 1, 1, 64)
    ce, se = c[..., 0::2], sn[..., 0::2]
    co, so = c[..., 1::2], sn[..., 1::2]
    rope[..., 448:512:2] = even * ce - odd * se
    rope[..., 449:512:2] = even * so + odd * co
    a = rope.to(x.dtype).reshape(s, 8, 4096).double()

    c_flat = torch.zeros((s, 8, 1024), dtype=torch.float64)
    for g in range(8):
        for block in range(4):
            k = torch.arange(block * 1024, (block + 1) * 1024)
            c_flat[:, g, :] += a[:, g, k] * w1[g, k, k % 1024].double()
    c_flat = c_flat.to(x.dtype).reshape(s, 8192).double()

    y = torch.zeros((s, 5120), dtype=torch.float64)
    for block in range(2):
        q = torch.arange(block * 5120, min((block + 1) * 5120, 8192))
        y[:, q % 5120] += c_flat[:, q] * w2[q % 5120, q].double()
    return y.to(x.dtype)


def sample(torch, torch_npu, *, S=8, seed=20261118, scale=0.05):
    rng = torch.Generator().manual_seed(seed + S)
    x = (torch.randn((S, 1, 64, 512), generator=rng) * 0.2).to(torch.bfloat16)
    cos = torch.randn((S, 1, 1, 64), generator=rng)
    sin = torch.randn((S, 1, 1, 64), generator=rng)

    w1 = torch.zeros((8, 4096, 1024), dtype=torch.bfloat16)
    for g in range(8):
        for block in range(4):
            k = torch.arange(block * 1024, (block + 1) * 1024)
            w1[g, k, k % 1024] = (torch.randn(1024, generator=rng) * scale).to(torch.bfloat16)
    w2 = torch.zeros((5120, 8192), dtype=torch.bfloat16)
    for block in range(2):
        q = torch.arange(block * 5120, min((block + 1) * 5120, 8192))
        w2[q % 5120, q] = (torch.randn(q.numel(), generator=rng) * scale).to(torch.bfloat16)

    expected = golden(torch, x, cos, sin, w1, w2)
    def device(tensor):
        return torch_npu.npu_format_cast(tensor.npu(), 2)

    return (device(x), device(cos), device(sin), device(w1), device(w2)), {"backend": "ascendc"}, expected


CASE_PARAMETERS = (
    dict(),
    dict(S=1),
    dict(S=16),
    dict(S=33),
    dict(S=96),
    dict(S=97),
    dict(S=8, scale=0.1),
)
# Common testing unpacks (atol, rtol) for each output.
TOLERANCES = ((0.015625, 0.015625),)
