"""BF16 inputs and an independent FP64 CPU reference for MatmulBiasSilu."""


CASE_PARAMETERS = (dict(m=16, k=48, n=32), dict(m=64, k=32, n=48))
TOLERANCES = ((0.008, 0.008),)


def sample(torch, torch_npu, *, m, k, n):
    rng = torch.Generator().manual_seed(20261026)
    a = (torch.randn(m, k, generator=rng) * 0.25).bfloat16()
    b = (torch.randn(k, n, generator=rng) * 0.25).bfloat16()
    bias = (torch.randn(n, generator=rng) * 0.1).bfloat16()
    expected = torch.nn.functional.silu(a.double() @ b.double() + bias.double())
    return (a.npu(), b.npu(), bias.npu()), {}, expected
