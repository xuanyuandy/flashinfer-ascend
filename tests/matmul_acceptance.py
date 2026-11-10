"""Fresh-process real Matmul checks; no numerical or execution backend is mocked."""

import json
import os
from pathlib import Path
import sys


def run(mode):
    import torch
    import torch_npu

    if "FLASHINFER_TEST_NPU_DEVICE" in os.environ:
        torch.npu.set_device(int(os.environ["FLASHINFER_TEST_NPU_DEVICE"]))

    import flashinfer
    from flashinfer._matmul import AOT_KERNEL_ID, TRITON_KERNEL_ID
    from flashinfer.jit import AscendCBackend, AscendCToolchain, TritonBackend
    from flashinfer.jit.discovery import AotCorruptError, AotMissingError, AotPackageIndex
    from flashinfer.runtime import get_global_registry

    assert torch.npu.is_available(), "a real NPU is required; skipping is not acceptance"
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    if mode.startswith("queue"):
        torch_npu.npu.set_task_queue_enable(int(mode[-1]))
    counts = dict(ascendc=0, triton=0, toolchain=0)
    expected_compiles = int(os.environ.get(
        "FLASHINFER_MATMUL_EXPECT_TRITON_COMPILES", "0" if mode == "cached_jit" else "1"))

    def forbidden(*args, **kwargs):
        raise AssertionError("AOT/cache execution attempted compilation or compiler discovery")

    def no_ascendc(*args, **kwargs):
        counts["ascendc"] += 1
        forbidden()

    def no_toolchain(*args, **kwargs):
        counts["toolchain"] += 1
        forbidden()

    compile_triton = TritonBackend.compile

    def jit(self, spec, *, build_directory):
        counts["triton"] += 1
        assert expected_compiles == 1, "a warm execution must not compile"
        return compile_triton(self, spec, build_directory=build_directory)

    AscendCBackend.compile = no_ascendc
    AscendCToolchain.discover = no_toolchain
    TritonBackend.compile = jit
    registry = get_global_registry()
    native = registry.get(AOT_KERNEL_ID).runner
    triton = registry.get(TRITON_KERNEL_ID).runner
    assert native.jit_backend_factory is None
    if mode in {"aot_without_compiler", "cached_jit"}:
        import subprocess
        subprocess.Popen = forbidden

    errors, pointers = [], []
    shapes = [(16, 16, 16), (32, 64, 48), (80, 48, 32), (128, 128, 128),
              (256, 256, 256), (256, 16, 256), (16, 256, 16)] if mode == "matrix" else [(32, 64, 48)]
    backends = ("triton",) if mode in {"cold_jit", "cached_jit"} else (
        ("ascendc", "cann") if mode == "aot_without_compiler" else ("cann", "ascendc", "triton"))
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for m, n, k in shapes:
            # Preserve positive storage offsets, and cross the TorchNPU host
            # queue before direct Triton launch, using the same tensors/out.
            ca = (torch.randn(m + 1, k) * .25).bfloat16()
            cb = (torch.randn(k + 1, n) * .25).bfloat16()
            a, b = ca.npu()[1:], cb.npu()[1:]
            a.add_(.125)
            expected = (ca[1:] + .125).double() @ cb[1:].double()
            out = torch.empty((m + 1, n), dtype=a.dtype, device=a.device)[1:]
            assert a.storage_offset() and b.storage_offset() and out.storage_offset()
            original_pointers = (a.data_ptr(), b.data_ptr(), out.data_ptr())
            if mode.startswith("missing") or mode == "corrupt":
                isolated = Path(os.environ["FLASHINFER_MATMUL_ERROR_ROOT"])
                native.index = AotPackageIndex(roots=(isolated,))
                try:
                    flashinfer.matmul(a, b, out=out, backend="ascendc")
                except (AotMissingError, AotCorruptError) as error:
                    assert isinstance(error, AotCorruptError if mode == "corrupt" else AotMissingError)
                    assert counts == dict(ascendc=0, triton=0, toolchain=0)
                    assert native.provenance is None
                    print(json.dumps(dict(result="PASS", mode=mode, category=error.category,
                                          fallback=False, compiler_calls=counts,
                                          diagnostic=str(error)), sort_keys=True), flush=True)
                    return
                raise AssertionError("expected explicit AOT failure")
            for backend in backends:
                for supply_output in (False, True):
                    result = flashinfer.matmul(a, b, out=out if supply_output else None, backend=backend)
                    assert result.shape == (m, n) and result.dtype == torch.bfloat16
                    if supply_output:
                        assert result is out
                    assert (a.data_ptr(), b.data_ptr(), out.data_ptr()) == original_pointers
                    torch.testing.assert_close(result.cpu().double(), expected, atol=.008, rtol=.008)
                    # A subsequent TorchNPU consumer must observe completed writes.
                    torch.testing.assert_close((result + .25).cpu(), expected.bfloat16() + .25,
                                               atol=.016, rtol=.008)
                    errors.append(dict(backend=backend, shape=[m, n, k], out=supply_output,
                                       max_abs=(result.cpu().double() - expected).abs().max().item()))
                pointers.append(dict(backend=backend, shape=[m, n, k], pointers=original_pointers))
            torch.testing.assert_close(a.cpu(), ca[1:] + .125, atol=0, rtol=0)
            torch.testing.assert_close(b.cpu(), cb[1:], atol=0, rtol=0)
    if "ascendc" in backends:
        assert native.provenance.source == "aot"
    assert counts["ascendc"] == counts["toolchain"] == 0
    if "triton" in backends:
        assert counts["triton"] == expected_compiles
        assert triton.build_requests == 1 and triton.launches == 2 * len(shapes)
    print(json.dumps(dict(result="PASS", mode=mode, compiler_calls=counts,
                          native_source=native.provenance.source if native.provenance else None,
                          triton_build_requests=triton.build_requests, errors=errors,
                          shared_pointers=pointers), sort_keys=True), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
