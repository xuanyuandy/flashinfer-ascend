"""Run the installed FlashAttention operator: python3 flash_attention.py."""

import os
from pathlib import Path
import sys


def main():
    # A wheel smoke test must use its bundled kernel, without compiling at run time.
    os.environ["FLASHINFER_NPU_JIT_DISABLE"] = "1"

    import torch
    import torch_npu  # noqa: F401
    import flashinfer
    from flashinfer.aclnn._ops import ensure_registered

    location = Path(flashinfer.__file__).resolve()
    print("flashinfer:", location, flush=True)
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Run this script from tests/ after installing the wheel in this environment")
    if not torch.npu.is_available():
        raise RuntimeError("FlashAttention requires an available Ascend NPU")

    ensure_registered()
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(20261026)
        cpu = tuple((torch.rand(shape) * 2 - 1).to(dtype) for shape in
                    ((128, 2, 128), (256, 2, 128), (256, 2, 128)))
        q, k, v = (tensor.npu() for tensor in cpu)
        output = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)
        supplied = torch.empty_like(output)
        assert flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=supplied) is supplied
        query, key, value = (tensor.double().transpose(0, 1) for tensor in cpu)
        probability = torch.softmax((query @ key.transpose(1, 2)) / (128 ** 0.5), dim=-1)
        expected = (probability @ value).transpose(0, 1)
        atol, rtol = (0.001, 0.001) if dtype == torch.float16 else (0.004, 0.008)
        for result in (output, supplied):
            assert result.dtype == dtype and result.shape == q.shape
            torch.testing.assert_close(result.cpu().double(), expected, atol=atol, rtol=rtol)
        print("output:", tuple(output.shape), output.dtype)
        print("max_abs:", (output.cpu().double() - expected).abs().max().item())
    print("PASS flash_attention")


if __name__ == "__main__":
    main()
