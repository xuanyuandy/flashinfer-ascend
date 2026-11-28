"""Independent CPU reference for the fused residual mix and coefficient update."""

HC_NORM_EPS = 2e-5
HC_PRE_EPS = 3e-4
HC_POST_SCALE = 1.25
SINKHORN_EPS = 2e-6
NUM_SINKHORN_ITERS = 10
RMSNORM_EPS = 7e-6


def golden(torch, x, residual, previous_mix, post_mix, comb_res_mix, fn,
           mix_scales, mix_bases, rmsnorm_weight):
    scales = torch.cat((mix_scales[0].expand(4), mix_scales[1].expand(4),
                        mix_scales[2].expand(16)))
    pieces = []
    for start in range(0, x.shape[0], 512):
        stop = min(start + 512, x.shape[0])
        r = (torch.einsum("tij,tih->tjh", comb_res_mix[start:stop].float(),
                          residual[start:stop].float())
             + x[start:stop].float().unsqueeze(1) * post_mix[start:stop].float())
        r = r.bfloat16()
        u = (r.float() * previous_mix[start:stop].float().unsqueeze(-1)).sum(1)
        u = u.bfloat16()
        flat = r.float().flatten(1)
        logits = (flat @ fn.float().T) * torch.rsqrt(
            flat.square().mean(1) + HC_NORM_EPS).unsqueeze(1)
        logits = logits * scales + mix_bases
        new_pre = logits[:, :4].sigmoid() + HC_PRE_EPS
        new_post = (logits[:, 4:8].sigmoid() * HC_POST_SCALE).unsqueeze(-1)
        new_comb = logits[:, 8:].reshape(-1, 4, 4).softmax(-1) + SINKHORN_EPS
        new_comb = new_comb / (new_comb.sum(-2, keepdim=True) + SINKHORN_EPS)
        for _ in range(NUM_SINKHORN_ITERS - 1):
            new_comb = new_comb / (new_comb.sum(-1, keepdim=True) + SINKHORN_EPS)
            new_comb = new_comb / (new_comb.sum(-2, keepdim=True) + SINKHORN_EPS)
        y = (u.float() * torch.rsqrt(
            u.float().square().mean(1) + RMSNORM_EPS).unsqueeze(1)
            * rmsnorm_weight.float()).bfloat16()
        pieces.append((r, new_post, new_comb, y, new_pre))
    return tuple(torch.cat(column) for column in zip(*pieces))


CASE_PARAMETERS = tuple(
    dict(t=t, h=h)
    for t, h in ((96, 5120), (96, 1024), (96, 2048), (128, 3072), (200, 4096),
                 (96, 6144), (96, 7168), (1, 1024), (1, 7168), (31, 4096),
                 (33, 4096), (257, 4096), (1025, 7168), (4096, 7168))
)
TOLERANCES = ((0.02, 0.01),) * 5


def sample(torch, torch_npu, *, t=96, h=5120):
    rng = torch.Generator().manual_seed(20261128 + t + h)
    x, residual, rmsnorm_weight = (
        torch.randn(shape, generator=rng).to(torch.bfloat16)
        for shape in ((t, h), (t, 4, h), (h,))
    )
    previous_mix, post_mix, comb_res_mix = (
        torch.rand(shape, generator=rng)
        for shape in ((t, 4), (t, 4, 1), (t, 4, 4))
    )
    fn = torch.rand((24, 4 * h), generator=rng) * 0.1 - 0.05
    mix_scales = torch.rand((3,), generator=rng) * 0.2 - 0.1
    mix_bases = torch.rand((24,), generator=rng) * 0.2 - 0.1
    inputs = (x, residual, previous_mix, post_mix, comb_res_mix, fn, mix_scales,
              mix_bases, rmsnorm_weight)
    expected = golden(torch, *inputs)
    npu_inputs = tuple(
        torch_npu.npu_format_cast(tensor.npu(), 2) for tensor in inputs
    )
    return npu_inputs, {}, expected
