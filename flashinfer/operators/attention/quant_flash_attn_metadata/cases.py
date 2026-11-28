"""Pure-Python golden and public samples for C4A4 metadata."""

from itertools import accumulate

from ._cpu_golden import forward_metadata


def golden(torch, *, dim, heads, kv_heads, q_lengths, kv_lengths, used_q, qscale_layout, mask_mode, aic, aiv):
    forward = forward_metadata(
        dim=dim, heads=heads, kv_heads=kv_heads, q_lengths=q_lengths, kv_lengths=kv_lengths,
        used_q=used_q, qscale_layout=qscale_layout, mask_mode=mask_mode, aic=aic, aiv=aiv,
    )
    elements = 16 * (1 + 108 * len(q_lengths) * heads)
    capacity = (elements + 4095) // 4096 * 4096
    metadata = torch.zeros((1, capacity), dtype=torch.int32)
    metadata[0, :len(forward)] = torch.tensor(forward, dtype=torch.int32)
    return metadata


def sample(torch, torch_npu, *, dim, q_lengths, kv_lengths, heads=1, kv_heads=1,
           used_q=None, qscale_layout="TND", kv_layout="PA_BBND", mask_mode=0):
    device = torch.device("npu", torch.npu.current_device())
    props = torch.npu.get_device_properties(device.index)
    expected = golden(
        torch, dim=dim, heads=heads, kv_heads=kv_heads, q_lengths=q_lengths, kv_lengths=kv_lengths,
        used_q=used_q, qscale_layout=qscale_layout, mask_mode=mask_mode,
        aic=props.cube_core_num, aiv=props.vector_core_num,
    )
    options = dict(
        cu_seqlens_q=torch.tensor((0, *accumulate(q_lengths)), dtype=torch.int32, device=device),
        seqused_kv=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        layout_q_descale=qscale_layout, layout_kv=kv_layout, mask_mode=mask_mode,
    )
    if used_q is not None:
        options["seqused_q"] = torch.tensor(used_q, dtype=torch.int32, device=device)
    return (heads, kv_heads, dim), options, expected


def comparison_mask(_output_index, reference):
    import torch

    sections = int(reference[0, 0])
    aic, aiv = reference[0, 4:6].tolist()
    initialized = 16 * (1 + sections * (aic + aiv))
    assert sections > 0 and aic > 0 and aiv > 0 and initialized <= reference.shape[1]
    mask = torch.zeros_like(reference, dtype=torch.bool)
    mask[0, :initialized] = True
    return mask


CASE_PARAMETERS = (
    dict(dim=128, q_lengths=(1,), kv_lengths=(1,)),
    dict(dim=256, q_lengths=(1,), kv_lengths=(1,)),
    dict(dim=128, heads=2, q_lengths=(3, 3), kv_lengths=(7, 7)),
    dict(dim=256, heads=4, kv_heads=2, q_lengths=(2, 2), kv_lengths=(7, 7),
         qscale_layout="N2TGD", kv_layout="PA_BNBD"),
    dict(dim=128, heads=4, kv_heads=2, q_lengths=(2,), kv_lengths=(4,),
         qscale_layout="N2TGD", kv_layout="PA_NZ"),
    dict(dim=256, q_lengths=(256,), used_q=(128,), kv_lengths=(9,), kv_layout="PA_BNBD"),
    dict(dim=128, q_lengths=(4,), kv_lengths=(2,), kv_layout="PA_NZ", mask_mode=3),
    dict(dim=256, heads=2, q_lengths=(2, 0, 2), kv_lengths=(7, 7, 7), kv_layout="PA_BNBD"),
    dict(dim=128, heads=4, kv_heads=2, q_lengths=(2, 2), kv_lengths=(0, 0),
         qscale_layout="N2TGD", kv_layout="PA_NZ"),
    dict(dim=128, q_lengths=(256,), kv_lengths=(129,)),
    dict(dim=128, heads=4, q_lengths=(17,), kv_lengths=(257,)),
    dict(dim=256, heads=8, q_lengths=(1, 1), kv_lengths=(5, 5),
         qscale_layout="N2TGD", kv_layout="PA_NZ"),

    dict(dim=128, heads=32, kv_heads=4, q_lengths=(17, 33), kv_lengths=(257, 513), kv_layout="PA_BNBD"),
    dict(dim=256, heads=32, kv_heads=2, q_lengths=(3,), kv_lengths=(4096,), kv_layout="PA_NZ"),
    dict(dim=256, heads=8, kv_heads=4, q_lengths=(129, 145, 514, 130),
         kv_lengths=(65536, 65536, 65536, 65536), kv_layout="PA_NZ"),
)
TOLERANCES = ((0, 0),)
