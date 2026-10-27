"""Run the installed MatmulBiasSiLU operator: python3 matmul_bias_silu.py."""

import os
from pathlib import Path
import sys


def main():
    os.environ["FLASHINFER_NPU_JIT_DISABLE"] = "1"

    import torch
    import torch_npu  # noqa: F401
    import flashinfer

    location = Path(flashinfer.__file__).resolve()
    print("flashinfer:", location, flush=True)
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Run this script from tests/ after installing the wheel in this environment")
    if not torch.npu.is_available():
        raise RuntimeError("MatmulBiasSiLU requires an available Ascend NPU")

    torch.manual_seed(20261026)
    a = (torch.randn(16, 48) * 0.25).bfloat16()
    b = (torch.randn(48, 32) * 0.25).bfloat16()
    bias = (torch.randn(32) * 0.1).bfloat16()
    output = flashinfer.matmul_bias_silu(a.npu(), b.npu(), bias.npu())

    expected = torch.nn.functional.silu(a.double() @ b.double() + bias.double())
    actual = output.cpu().double()
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    print("output:", tuple(output.shape), output.dtype)
    print("max_abs:", (actual - expected).abs().max().item())
    print("PASS matmul_bias_silu")


if __name__ == "__main__":
    main()
