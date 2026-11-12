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
    weights = safe.softmax(1)
    output = (v.float() * weights[..., None]).sum(1).to(v.dtype)
    lse = safe.logsumexp(1) / math.log(2)
    return (torch.where(valid[..., None], output, 0), lse.masked_fill(~valid, -math.inf))
