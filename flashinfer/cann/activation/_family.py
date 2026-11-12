"""Activation implementations; torch and torch_npu are imported at execution."""
from functools import partial
NAMES = ('silu_and_mul', 'gelu_and_mul', 'gelu_tanh_and_mul')

def execute(name, input, out):
    import torch
    if input.numel() == 0:
        return out
    import torch_npu
    if input.device.type != 'npu':
        raise ValueError('CANN activation requires NPU tensors')
    if name == 'silu_and_mul':
        result = torch_npu.npu_swiglu(input.reshape(-1, input.shape[-1]), dim=-1).reshape(out.shape)
    else:
        half = input.shape[-1] // 2
        flat = input.reshape(-1, 2 * half)
        result = torch.empty((*input.shape[:-1], half), dtype=input.dtype, device=input.device)
        dest = result.reshape(-1, half)
        for row in range(0, flat.shape[0], 200000):
            for col in range(0, half, 512):
                width = min(512, half - col)
                tile = torch.cat((flat[row:row + 200000, col:col + width], flat[row:row + 200000, half + col:half + col + width]), -1)
                dest[row:row + 200000, col:col + width] = torch_npu.npu_gelu_mul(tile, approximate='tanh' if name == 'gelu_tanh_and_mul' else 'none')
    out.copy_(result)
    return out
