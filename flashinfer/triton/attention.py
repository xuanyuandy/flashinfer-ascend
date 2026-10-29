"""Source-backed dense FA declaration and lazy, synchronous Triton runner."""

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import RLock

from ..runtime import SchemaError
from ..tuning.identity import runtime_identity


TRITON_FLASH_ATTENTION_KERNEL_ID = "triton/attention/dense_prefill_v1/Ascend950"
_ENTRY = "_dense_prefill_kernel"
_CONSTANTS = {"D": 128, "BLOCK_M": 32, "BLOCK_N": 64}
_OPTIONS = {"num_warps": 4, "num_stages": 1}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TritonFlashAttentionRunner:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self._modules = {}
        self._specs = {}
        self._lock = RLock()
        self.build_requests = 0
        self.launches = 0

    @lru_cache(maxsize=2)
    def _execution_identity(self, dtype):
        from ..tuning.identity import execution_sources

        return _digest(dict(spec=self.spec(dtype).fingerprint,
                            runner=execution_sources(("triton/attention.py", "jit/_triton_compile.py",
                                                      "jit/_triton_abi.py", "jit/loading.py")),
                            runtime=runtime_identity()))

    def tuning_identity(self, signature, hardware):
        return self._execution_identity(signature.dtypes[0])

    def spec(self, dtype):
        from ..jit import JitEnvironment, TritonBackend

        if dtype not in {"float16", "bfloat16"}:
            raise SchemaError("Triton FA requires float16 or bfloat16")
        environment = runtime_identity()
        pointer = "*fp16" if dtype == "float16" else "*bf16"
        signature = {"q": pointer, "k": pointer, "v": pointer, "out": pointer,
                     "S": "i32", "T": "i32", "H": "i32"}
        build_environment = JitEnvironment(
            "Ascend950", "Ascend950", _digest(environment["cann"]), "triton-ascend",
            environment["triton"], environment["torch"], environment["torch_npu"],
            environment["python_abi"],
        )
        return TritonBackend().declare(
            self.descriptor, target_soc="Ascend950", output_path="attention/kernel.bin",
            entry_point=_ENTRY, environment_fingerprint=build_environment.fingerprint,
            specialization_fingerprint=_digest(dict(dtype=dtype, constants=_CONSTANTS, options=_OPTIONS)),
            compile_options=("signature=" + json.dumps(signature, sort_keys=True),
                             "constexprs=" + json.dumps(_CONSTANTS, sort_keys=True),
                             "options=" + json.dumps(_OPTIONS, sort_keys=True),
                             'binary={"mode":"mix"}'),
        )

    def prepare(self, dtype, device_index):
        from ..jit import TritonBackend, build_and_load

        key = (dtype, device_index)
        with self._lock:
            if key not in self._modules:
                root = Path(__file__).resolve().parents[1]
                source = root / self.descriptor.source.locator
                if hashlib.sha256(source.read_bytes()).hexdigest() != self.descriptor.source.digest:
                    raise SchemaError("Triton FA SourceRef digest mismatch; refresh the declaration")
                spec = self.spec(dtype)
                module = build_and_load(spec, backend=TritonBackend())
                self._modules[key] = module
                self._specs[key] = spec
                self.build_requests += 1
            return self._modules[key]

    def __call__(self, q, k, v, out):
        import torch
        import torch_npu
        from ..ascendc import FLASH_ATTENTION_KERNEL_ID, tensor_contract_specialization
        from ..jit.tensor_adapter import _reject_output_overlap
        from ..ascendc import load_contract

        tensors = (q, k, v, out)
        contract = tensor_contract_specialization(FLASH_ATTENTION_KERNEL_ID, tensors)
        if contract != "supported":
            raise ValueError(contract)
        _, recipe, _ = load_contract(FLASH_ATTENTION_KERNEL_ID)
        _reject_output_overlap(recipe.tensor_parameters, tensors)
        if q.device.type != torch._C._get_privateuse1_backend_name():
            raise ValueError("Triton FA requires NPU tensors")
        if any(torch_npu.get_npu_format(tensor) != 2 for tensor in tensors):
            raise ValueError("Triton FA requires contiguous ND tensors")
        with torch.npu.device(q.device):
            module = self.prepare(str(q.dtype).removeprefix("torch."), torch.npu.current_device())
            stream = torch.npu.current_stream(q.device)
            # This public FA contract is synchronous eager. Drain prior TorchNPU
            # host submissions before a direct Triton launch on the same stream.
            stream.synchronize()
            for tensor in tensors:
                tensor.record_stream(stream)
            s, heads, _ = q.shape
            try:
                module.opaque_module[((s + _CONSTANTS["BLOCK_M"] - 1) // _CONSTANTS["BLOCK_M"], heads)](
                    q, k, v, out, s, k.shape[0], heads, stream=stream.npu_stream)
                self.launches += 1
            finally:
                stream.synchronize()


@lru_cache(maxsize=1)
def attention_descriptors():
    from ..runtime.manifest import load_kernel_manifest

    path = Path(__file__).resolve().parents[1] / "data/kernels/triton_registry.json"
    return tuple(replace(descriptor, runner=TritonFlashAttentionRunner(descriptor))
                 for descriptor in load_kernel_manifest(path))


def flash_attention_registry(base=None):
    """Opt-in registry for FA experiments; never extend the production singleton."""
    from ..runtime import Backend, KernelRegistry, get_global_registry

    base = get_global_registry() if base is None else base
    registry = KernelRegistry()
    for descriptor in base.descriptors() + attention_descriptors():
        if descriptor.backend == Backend.CUSTOM:
            registry.register_custom(descriptor)
        elif descriptor.backend == Backend.EXTERNAL:
            registry.register_external(descriptor)
        else:
            registry.register(descriptor)
    registry.freeze()
    return registry
