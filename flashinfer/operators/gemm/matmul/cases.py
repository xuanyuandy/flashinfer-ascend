"""BF16 inputs and an independent FP64 CPU reference for public Matmul."""


CASE_PARAMETERS = (dict(m=32, k=48, n=64), dict(m=16, k=256, n=16))
TOLERANCES = ((0.008, 0.008),)


def sample(torch, torch_npu, *, m, k, n):
    rng = torch.Generator().manual_seed(20261109)
    a = (torch.randn(m, k, generator=rng) * 0.25).bfloat16()
    b = (torch.randn(k, n, generator=rng) * 0.25).bfloat16()
    return (a.npu(), b.npu()), {}, a.double() @ b.double()
