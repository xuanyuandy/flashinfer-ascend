"""Numerical acceptance of the local chip's packaged CATLASS prefill."""

import json


def main():
    import torch
    import torch_npu  # noqa: F401
    import flashinfer
    from flashinfer.runtime import current_hardware

    assert torch.npu.is_available(), "CATLASS numerical acceptance requires an NPU"
    torch.set_num_threads(4)
    target = current_hardware().soc_model
    assert target in {"Ascend910B", "Ascend910C", "Ascend950"}
    kernel_id = "ascendc/t4/flash_attention/catlass/op_package/" + target
    errors = []
    stream = torch.npu.Stream()
    for dtype in (torch.float16, torch.bfloat16):
        for s, t, h in ((128, 128, 1), (128, 256, 2), (256, 384, 4),
                        (384, 512, 3), (512, 512, 8), (512, 128, 1)):
            torch.manual_seed(123)
            cpu = tuple((torch.rand(z) * 2 - 1).to(dtype)
                        for z in ((s, h, 128), (t, h, 128), (t, h, 128)))
            a, b, c = (x.double().transpose(0, 1) for x in cpu)
            expected = (torch.softmax(a @ b.transpose(1, 2) / 128**.5, -1) @ c).transpose(0, 1)
            # Inputs, both public overloads and output writes use the caller's stream.
            with torch.npu.stream(stream):
                q, k, v = (x.npu() for x in cpu)
                actual = flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, kernel_id=kernel_id)
                supplied = torch.empty_like(actual)
                assert flashinfer.single_prefill_with_kv_cache_catlass(
                    q, k, v, out=supplied, kernel_id=kernel_id) is supplied
            stream.synchronize()
            atol, rtol = (.001, .001) if dtype == torch.float16 else (.004, .008)
            for result in (actual, supplied):
                torch.testing.assert_close(result.cpu().double(), expected, atol=atol, rtol=rtol)
            errors.append(dict(dtype=str(dtype), shape=[s, t, h],
                               max_abs=float((actual.cpu().double() - expected).abs().max())))
    # A constant V must survive normalization; this also catches accidental
    # paged-cache indexing in the dense implementation.
    q = torch.ones((128, 1, 128), dtype=torch.float16, device="npu")
    result = flashinfer.single_prefill_with_kv_cache_catlass(q, q, q)
    torch.testing.assert_close(result.cpu(), torch.ones_like(q.cpu()), atol=.001, rtol=.001)
    print(json.dumps(dict(result="PASS", target=target, errors=errors), sort_keys=True))


if __name__ == "__main__":
    main()
