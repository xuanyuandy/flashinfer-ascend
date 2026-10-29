"""Exercise Python/Torch production interfaces through the native ACLNN bridge."""

import json
import sys


def run(case, dtype="float16"):
    import torch
    import torch_npu  # noqa: F401
    import flashinfer
    from flashinfer.aclnn import _native
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID
    from flashinfer.jit.op_package import OpPackageModule
    from flashinfer.runtime import Backend

    def forbidden(*args, **kwargs):
        raise AssertionError("execution bypassed the native Torch bridge")

    OpPackageModule.get_workspace_size = forbidden
    OpPackageModule.launch = forbidden
    torch.manual_seed(20261027)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        if case == "matmul":
            ac = (torch.randn(17, 48) * .25).bfloat16()
            bc = (torch.randn(49, 32) * .25).bfloat16()
            biasc = (torch.randn(33) * .1).bfloat16()
            expected = torch.nn.functional.silu(ac[1:].double() @ bc[1:].double() + biasc[1:].double())
            a, b, bias = ac.npu()[1:], bc.npu()[1:], biasc.npu()[1:]
            python_result = flashinfer.matmul_bias_silu(
                a, b, bias, backend=Backend.ASCENDC, kernel_id=MATMUL_BIAS_SILU_KERNEL_ID)
            result = torch.ops.flashinfer.matmul_bias_silu(
                a, b, bias, backend="ascendc", kernel_id=MATMUL_BIAS_SILU_KERNEL_ID)
            out = torch.empty_like(result)
            assert torch.ops.flashinfer.matmul_bias_silu_out(a, b, bias, out) is None
            assert flashinfer.matmul_bias_silu(a, b, bias, out=out) is out
            atol = rtol = .008
        elif case == "attention":
            dtype = getattr(torch, dtype)
            qc = (torch.rand(129, 2, 128) * 2 - 1).to(dtype)
            kc = (torch.rand(257, 2, 128) * 2 - 1).to(dtype)
            vc = (torch.rand(257, 2, 128) * 2 - 1).to(dtype)
            query, key, value = (t[1:].double().transpose(0, 1) for t in (qc, kc, vc))
            expected = (torch.softmax(query @ key.transpose(1, 2) / 128**.5, -1) @ value).transpose(0, 1)
            q, k, v = qc.npu()[1:], kc.npu()[1:], vc.npu()[1:]
            python_result = flashinfer.single_prefill_with_kv_cache_catlass(
                q, k, v, backend=Backend.ASCENDC, kernel_id=FLASH_ATTENTION_KERNEL_ID)
            result = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(
                q, k, v, backend="ascendc", kernel_id=FLASH_ATTENTION_KERNEL_ID)
            out = torch.empty_like(result)
            assert torch.ops.flashinfer.single_prefill_with_kv_cache_catlass_out(q, k, v, out) is None
            assert flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=out) is out
            atol, rtol = (.001, .001) if dtype == torch.float16 else (.004, .008)
        else:
            raise ValueError(case)
    for tensor in (python_result, result, out):
        torch.testing.assert_close(tensor.cpu().double(), expected, atol=atol, rtol=rtol)
    assert len(_native._owners) == 1
    assert torch._C._dispatch_has_kernel_for_dispatch_key("flashinfer_aclnn::_execute", "PrivateUse1")
    assert result.dtype == out.dtype
    assert flashinfer.norm._torch_library_ops is None
    print(json.dumps(dict(case=case, dtype=str(result.dtype), result="PASS", adapter="C++/OpCommand",
                          nondefault_stream=True, nonzero_storage_offset=True,
                          max_abs=float((result.cpu().double() - expected).abs().max()))))


if __name__ == "__main__":
    run(*sys.argv[1:])
