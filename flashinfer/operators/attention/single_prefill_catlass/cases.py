"""Dense unmasked NHD prefill goldens shared by all CATLASS target SoCs."""


CASE_PARAMETERS = tuple(
    dict(dtype=dtype, qlen=qlen, klen=klen)
    for dtype, lengths in (("float16", ((128, 256), (128, 128), (256, 512))),
                           ("bfloat16", ((256, 128), (128, 128), (128, 256), (256, 512))))
    for qlen, klen in lengths
)
TOLERANCES = ((0.004, 0.008),)


def tolerances(configuration):
    return ((0.001, 0.001),) if configuration["dtype"] == "float16" else TOLERANCES


def sample(torch, torch_npu, *, dtype, qlen, klen):
    rng = torch.Generator().manual_seed(20261026)
    q, k, v = ((torch.rand(length, 2, 128, generator=rng) * 2 - 1).to(getattr(torch, dtype))
               for length in (qlen, klen, klen))
    query, key, value = (tensor.double().transpose(0, 1) for tensor in (q, k, v))
    probabilities = torch.softmax((query @ key.transpose(1, 2)) / 128**0.5, dim=-1)
    return (q.npu(), k.npu(), v.npu()), {}, (probabilities @ value).transpose(0, 1)
