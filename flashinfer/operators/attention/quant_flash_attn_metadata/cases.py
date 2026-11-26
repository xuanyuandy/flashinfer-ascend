"""One FA task with one query token, one KV token and one head."""


def sample(torch, torch_npu, *, dim):
    device = torch.device("npu", torch.npu.current_device())
    props = torch.npu.get_device_properties(device.index)
    aic, aiv = props.cube_core_num, props.vector_core_num
    golden = torch.zeros((2, 4096), dtype=torch.int32)
    golden[0, :6] = torch.tensor((1, 0, 128, 256 if dim == 128 else 128, aic, aiv), dtype=torch.int32)
    golden[0, 19] = 1
    options = dict(cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
                   seqused_kv=torch.tensor([1], dtype=torch.int32, device=device))
    return (1, 1, dim), options, golden


def comparison_mask(output_index, reference):
    # Only the forward schedule is defined for inference. Clear() writes the
    # header and one section of FA and FD records in these cases.
    import torch

    aic, aiv = reference[0, 4:6].tolist()
    mask = torch.zeros_like(reference, dtype=torch.bool)
    mask[0, :16 * (1 + aic + aiv)] = True
    return mask


CASE_PARAMETERS = (dict(dim=128), dict(dim=256))
TOLERANCES = ((0, 0),)
