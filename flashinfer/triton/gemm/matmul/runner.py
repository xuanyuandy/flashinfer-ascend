from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import RLock
from flashinfer._matmul import TRITON_KERNEL_ID, validate_tensors
from flashinfer.runtime import Backend, CompileMode, KernelDescriptor, SchemaError, SourceRef
from flashinfer.tuning.identity import execution_sources, fingerprint, runtime_identity
_CONSTANTS = dict(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
_OPTIONS = dict(num_warps=4, num_stages=1)
_SOURCE_DIGEST = "61c4dc81d3e6737d972c17852b4f96d84cee1160d1ea8a7c4507f2ad70df37db"

class TritonMatmulRunner:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self._modules = {}
        self._lock = RLock()
        self.build_requests = 0
        self.launches = 0

    @lru_cache(maxsize=1)
    def spec(self):
        from flashinfer.jit import JitEnvironment, TritonBackend

        environment = runtime_identity()
        signature = dict(a="*bf16", b="*bf16", out="*bf16", M="i32", N="i32", K="i32")
        build_environment = JitEnvironment(
            "Ascend950", "Ascend950", fingerprint(environment["cann"]), "triton-ascend",
            environment["triton"], environment["torch"], environment["torch_npu"],
            environment["python_abi"],
        )
        return TritonBackend().declare(
            self.descriptor, target_soc="Ascend950", output_path="matmul/kernel.bin",
            entry_point="_matmul_kernel", environment_fingerprint=build_environment.fingerprint,
            specialization_fingerprint=fingerprint(dict(constants=_CONSTANTS, options=_OPTIONS)),
            compile_options=("signature=" + json.dumps(signature, sort_keys=True),
                             "constexprs=" + json.dumps(_CONSTANTS, sort_keys=True),
                             "options=" + json.dumps(_OPTIONS, sort_keys=True),
                             'binary={"mode":"mix"}'),
        )

    @lru_cache(maxsize=1)
    def tuning_identity(self, signature, hardware):
        return fingerprint(dict(spec=self.spec().fingerprint, runtime=runtime_identity(),
                                runner=execution_sources(("triton/gemm/matmul/runner.py", "_matmul.py",
                                                          "jit/_triton_compile.py", "jit/_triton_abi.py",
                                                          "jit/loading.py"))))

    def prepare(self, device_index):
        from flashinfer.jit import TritonBackend, build_and_load

        with self._lock:
            if device_index not in self._modules:
                source = Path(__file__).resolve().parents[3] / self.descriptor.source.locator
                if hashlib.sha256(source.read_bytes()).hexdigest() != self.descriptor.source.digest:
                    raise SchemaError("Triton Matmul SourceRef digest mismatch; refresh the declaration")
                self._modules[device_index] = build_and_load(self.spec(), backend=TritonBackend())
                self.build_requests += 1
            return self._modules[device_index]

    def __call__(self, a, b, out):
        import torch

        validate_tensors(a, b, out)
        with torch.npu.device(a.device):
            module = self.prepare(torch.npu.current_device())
            stream = torch.npu.current_stream(a.device)
            # Drain TorchNPU's host queue before submitting directly to its stream.
            stream.synchronize()
            for tensor in (a, b, out):
                tensor.record_stream(stream)
            m, k = a.shape
            n = b.shape[1]
            try:
                block_m, block_n = _CONSTANTS["BLOCK_M"], _CONSTANTS["BLOCK_N"]
                module.opaque_module[((m + block_m - 1) // block_m, (n + block_n - 1) // block_n)](
                    a, b, out, m, n, k, stream=stream.npu_stream)
                self.launches += 1
            finally:
                stream.synchronize()
