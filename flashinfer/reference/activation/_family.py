"""Activation implementations; torch and torch_npu are imported at execution."""
from functools import partial
NAMES = ('silu_and_mul', 'gelu_and_mul', 'gelu_tanh_and_mul')

def execute(name, input, out):
    import torch
    if input.numel() == 0:
        return out
    import torch.nn.functional as functional
    (a, b) = input.float().chunk(2, dim=-1)
    active = functional.silu(a) if name == 'silu_and_mul' else functional.gelu(a, approximate='tanh' if name == 'gelu_tanh_and_mul' else 'none')
    result = (active * b).to(input.dtype)
    out.copy_(result)
    return out
