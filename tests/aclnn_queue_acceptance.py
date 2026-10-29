"""Mixed TorchNPU/ACLNN execution, two packages and immutable warm bindings."""

import json
import sys


def run(mode, first):
    import torch
    import torch_npu
    import flashinfer
    from flashinfer.aclnn import enable_aclnn, _native
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID
    from flashinfer.runtime import get_global_registry

    torch_npu.npu.set_task_queue_enable(mode)
    assert torch_npu.npu.get_task_queue_enable() == mode
    enable_aclnn()
    torch.manual_seed(20261028)
    ac = (torch.randn(16, 48) * .2).bfloat16()
    bc = (torch.randn(48, 32) * .2).bfloat16()
    biasc = (torch.randn(32) * .05).bfloat16()
    qc = (torch.rand(128, 2, 128) * .4 - .2).half()
    kc = (torch.rand(256, 2, 128) * .4 - .2).half()
    vc = (torch.rand(256, 2, 128) * .4 - .2).half()
    expected_mm = torch.nn.functional.silu((ac + .125).double() @ bc.double() + biasc.double())
    query = (qc + .125).double().transpose(0, 1)
    key, value = (t.double().transpose(0, 1) for t in (kc, vc))
    expected_fa = (torch.softmax(query @ key.transpose(1, 2) / 128**.5, -1) @ value).transpose(0, 1)
    stream = torch.npu.Stream()
    observations = []
    with torch.npu.stream(stream):
        a, b, bias = ac.npu(), bc.npu(), biasc.npu()
        q, k, v = qc.npu(), kc.npu(), vc.npu()
        order = ("matmul", "attention") if first == "matmul" else ("attention", "matmul")
        for repeat in range(3):
            for case in order:
                if case == "matmul":
                    # These producers/consumers use the ordinary TorchNPU queue.
                    produced = a + .125
                    output = torch.ops.flashinfer.matmul_bias_silu(produced, b, bias)
                    consumed = output + .25
                    expected = expected_mm.bfloat16() + .25
                    atol = rtol = .008
                else:
                    produced = q + .125
                    output = flashinfer.single_prefill_with_kv_cache_catlass(produced, k, v)
                    consumed = output + .25
                    expected = expected_fa.half() + .25
                    atol = rtol = .001
                observations.append((case, consumed, expected, atol, rtol))
            if repeat == 0:
                assert len(_native._owners) == 2

                def forbidden(*args, **kwargs):
                    raise AssertionError("warm native calls must reuse prepared bindings")

                _native.prepare = forbidden
    errors = {}
    for case, observed, expected, atol, rtol in observations:
        actual = observed.cpu()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        errors[case] = max(errors.get(case, 0), float((actual.double() - expected.double()).abs().max()))
    registry = get_global_registry()
    assert {registry.get(key).runner.provenance.source for key in
            (FLASH_ATTENTION_KERNEL_ID, MATMUL_BIAS_SILU_KERNEL_ID)} == {"aot"}
    print(json.dumps(dict(result="PASS", task_queue_mode=mode, first=first,
                          packages=2, invocations=len(observations),
                          mixed_torchnpu=True, nondefault_stream=True, warm_rebinds=0,
                          max_abs=errors), sort_keys=True))


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2])
