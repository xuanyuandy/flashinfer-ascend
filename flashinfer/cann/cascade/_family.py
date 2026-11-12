"""Stacked state merge with FlashInfer's base-2 LSE contract."""
from functools import partial
import math

def merge_states_execute(v, s):
    import torch
    if v.ndim != 4 or s.shape != v.shape[:3] or v.shape[1] == 0:
        raise ValueError('merge_states expects v [seq, states > 0, heads, dim] and matching s')
    if s.device != v.device:
        raise ValueError('v and s must be on the same device')
    if bool((torch.isnan(s) | torch.isposinf(s)).any()):
        raise ValueError('merge_states supports finite or negative-infinite LSE')
    (seq, states, heads, dim) = v.shape
    valid = torch.isfinite(s).any(dim=1)
    natural = s.float() * math.log(2)
    safe = torch.where(valid[:, None, :], natural, torch.zeros_like(natural))
    if v.numel() == 0:
        return (v.new_empty((seq, heads, dim)), s.new_empty((seq, heads), dtype=torch.float32))
    import torch_npu
    if v.device.type != 'npu':
        raise ValueError('CANN merge_states requires NPU tensors')
    logs = [part.reshape(-1).contiguous() for part in natural.unbind(1)]
    outputs = [part.reshape(-1, dim).contiguous() for part in v.unbind(1)]
    if states > 15:
        outputs = [part.float() for part in outputs]

    def merge_group(group_logs, group_outputs):
        supported = torch.isfinite(torch.stack(group_logs)).any(0)
        safe_logs = [torch.where(supported, item, 0) for item in group_logs]
        (out, log) = torch_npu.npu_attention_update(safe_logs, group_outputs, 1)
        return (torch.where(supported[:, None], out, 0), log.masked_fill(~supported, -math.inf))
    while len(logs) > 15:
        merged = [merge_group(logs[i:i + 15], outputs[i:i + 15]) for i in range(0, len(logs), 15)]
        (outputs, logs) = ([x[0] for x in merged], [x[1] for x in merged])
    (output, lse) = merge_group(logs, outputs)
    output = output.reshape(seq, heads, dim).to(v.dtype)
    lse = lse.reshape(seq, heads) / math.log(2)
    return (torch.where(valid[..., None], output, 0), lse.masked_fill(~valid, -math.inf))
