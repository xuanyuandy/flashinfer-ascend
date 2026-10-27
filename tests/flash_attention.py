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

    location = Path(flashinfer.__file__).resolve()
    print("flashinfer:", location, flush=True)
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Run this script from tests/ after installing the wheel in this environment")
    if not torch.npu.is_available():
        raise RuntimeError("FlashAttention requires an available Ascend NPU")

    torch.manual_seed(20261026)
    q = (torch.rand(128, 2, 128) * 2 - 1).half()
    k = (torch.rand(256, 2, 128) * 2 - 1).half()
    v = (torch.rand(256, 2, 128) * 2 - 1).half()
    output = flashinfer.single_prefill_with_kv_cache_catlass(q.npu(), k.npu(), v.npu())

    query, key, value = (tensor.double().transpose(0, 1) for tensor in (q, k, v))
    probability = torch.softmax((query @ key.transpose(1, 2)) / (128 ** 0.5), dim=-1)
    expected = (probability @ value).transpose(0, 1)
    actual = output.cpu().double()
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.001)
    print("output:", tuple(output.shape), output.dtype)
    print("max_abs:", (actual - expected).abs().max().item())
    print("PASS flash_attention")


if __name__ == "__main__":
    main()
