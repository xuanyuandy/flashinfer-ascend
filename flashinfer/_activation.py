"""Activation implementations; torch and torch_npu are imported at execution."""

from functools import partial

NAMES = ("silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul")


def execute(name, input, out, *, native):
    import torch

    if input.numel() == 0:
        return out
    if native:
        import torch_npu

        if input.device.type != "npu":
            raise ValueError("CANN activation requires NPU tensors")
        if name == "silu_and_mul":
            result = torch_npu.npu_swiglu(input.reshape(-1, input.shape[-1]), dim=-1).reshape(out.shape)
        else:
            # Native GeluMul supports <=1024 input columns and <=200000 rows.
            # GELU is pointwise; tile both matching halves together so larger
            # hidden sizes preserve the same formula without an unfused path.
            half = input.shape[-1] // 2
            flat = input.reshape(-1, 2 * half)
            result = torch.empty((*input.shape[:-1], half), dtype=input.dtype, device=input.device)
            dest = result.reshape(-1, half)
            for row in range(0, flat.shape[0], 200000):
                for col in range(0, half, 512):
                    width = min(512, half - col)
                    tile = torch.cat((flat[row:row + 200000, col:col + width],
                                      flat[row:row + 200000, half + col:half + col + width]), -1)
                    dest[row:row + 200000, col:col + width] = torch_npu.npu_gelu_mul(
                        tile, approximate="tanh" if name == "gelu_tanh_and_mul" else "none",
                    )
    else:
        import torch.nn.functional as functional

        a, b = input.float().chunk(2, dim=-1)
        active = functional.silu(a) if name == "silu_and_mul" else functional.gelu(
            a, approximate="tanh" if name == "gelu_tanh_and_mul" else "none",
        )
        result = (active * b).to(input.dtype)
    out.copy_(result)
    return out


def descriptors(backend):
    from .runtime.schema import Backend, CapabilitySpec, KernelDescriptor

    native = backend == Backend.CANN
    return tuple(KernelDescriptor(
        kernel_id=f"{backend.value}/activation/{name}_v1", backend=backend,
        capability=CapabilitySpec(
            op="activation." + name, dtypes=("float16", "bfloat16", "float32"),
            layouts=("ND",), soc_models=("Ascend950",) if native else (),
        ), runner=partial(execute, name, native=native),
    ) for name in NAMES)
