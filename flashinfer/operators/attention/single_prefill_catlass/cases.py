"""Dense unmasked NHD prefill goldens shared by all CATLASS target SoCs."""


CASE_PARAMETERS = (dict(dtype="float16", qlen=128, klen=256),
                   dict(dtype="bfloat16", qlen=256, klen=128))
TOLERANCES = ((0.004, 0.008),)


def sample(torch, torch_npu, *, dtype, qlen, klen):
    rng = torch.Generator().manual_seed(20261026)
    q, k, v = ((torch.rand(length, 2, 128, generator=rng) * 2 - 1).to(getattr(torch, dtype))
               for length in (qlen, klen, klen))
    query, key, value = (tensor.double().transpose(0, 1) for tensor in (q, k, v))
    probabilities = torch.softmax((query @ key.transpose(1, 2)) / 128**0.5, dim=-1)
    return (q.npu(), k.npu(), v.npu()), {}, (probabilities @ value).transpose(0, 1)
