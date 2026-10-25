"""Explicit torch.library declarations and bridges for prepared Ascend C ops.

Import and declaration construction use no torch, SDK, compiler or device I/O.
Only enable_torch_library registers declarations; only device execution touches
torch_npu/CANN. BuildRecipe, build/store and loader contracts remain in T13.
"""

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Callable, Tuple


@dataclass(frozen=True)
class AscendCOp:
    """One schema, PrivateUse1 implementation, and metadata-only implementation.

    Callables must follow ``schema``. ``mutates_args`` names its write aliases;
    returned tensors must not alias arguments. The device callable owns its
    prepared module/recipe and ABI conversion. The metadata callable must use
    only tensor metadata, never values, runtime dispatch or device allocation.
    """

    name: str
    schema: str
    device_impl: Callable
    meta_impl: Callable
    mutates_args: Tuple[str, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.name, str)
                or not re.fullmatch(r"flashinfer::[a-z][a-z0-9_]*", self.name)
                or any(word in self.name.lower() for word in ("vllm", "_c_ascend"))):
            raise ValueError("Ascend C op names must be flashinfer:: names without product semantics")
        if self.name in {"flashinfer::rmsnorm", "flashinfer::fused_add_rmsnorm"}:
            raise ValueError("Ascend C op name is reserved by a builtin")
        if not isinstance(self.schema, str) or not self.schema.startswith("("):
            raise ValueError("schema must be an unnamed torch schema starting with '('")
        if not callable(self.device_impl) or not callable(self.meta_impl):
            raise TypeError("device_impl and meta_impl must be callable")
        mutations = tuple(self.mutates_args)
        if (any(not isinstance(arg, str) or not arg for arg in mutations)
                or len(set(mutations)) != len(mutations)):
            raise ValueError("mutates_args must contain unique argument names")
        object.__setattr__(self, "mutates_args", mutations)


def matmul_probe_op(recipe, module, *, spec=None):
    """Adapt a verified T13 module to ``flashinfer::matmul_probe(a, b, out)``.

    For host_shared, pass the JitLoadedModule and its JitSpec returned by T13.
    For op_package, pass load_op_package's OpPackageModule (before custom-op execution),
    or a JitLoadedModule plus spec. This function neither loads nor registers.
    Native/CATLASS selection stays entirely in the supplied recipe/config.
    """
    from flashinfer._meta.ascendc import matmul_probe_meta
    from flashinfer.runtime import SchemaError

    from .loading import JitLoadedModule
    from .op_package import OpPackageModule
    from .recipe import BuildRecipe

    if not isinstance(recipe, BuildRecipe) or recipe.recipe_id != "matmul_probe":
        raise SchemaError("matmul_probe_op requires a T13 matmul_probe BuildRecipe")
    if isinstance(module, JitLoadedModule):
        from .core import JitSpec

        if (not isinstance(spec, JitSpec) or spec.generator_id != "ascendc"
                or module.receipt.spec_fingerprint != spec.fingerprint
                or spec.kernel_id != recipe.kernel_id or spec.source != recipe.source
                or spec.target_soc != recipe.target_soc
                or spec.entry_points != (recipe.host_entry,)
                or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint
                or (recipe.output_kind == "op_package"
                    and dict(spec.input_fingerprints).get("output_adapter") != _adapter_hash(recipe))
                or tuple(s.name for s in module.receipt.symbols) != (recipe.host_entry,)):
            raise SchemaError("loaded Ascend C module/spec does not bind the recipe/adapter/entry")
        module = module.opaque_module
    elif recipe.output_kind == "host_shared":
        raise SchemaError("host_shared requires a verified JitLoadedModule and its spec")

    if recipe.output_kind == "host_shared":
        if recipe.launch_abi != "fi.matmul_probe.v1" or not hasattr(module, recipe.host_entry):
            raise SchemaError("host_shared does not implement the T13 probe ABI")
        import ctypes as C

        entry = getattr(module, recipe.host_entry)
        entry.argtypes, entry.restype = [C.c_void_p] * 4, None
    else:
        if (recipe.launch_abi != "aclnn.matmul_probe.v1"
                or not isinstance(module, OpPackageModule) or module.recipe != recipe):
            raise SchemaError("op_package does not bind the T13 probe recipe/ABI")
        entry = None
    return AscendCOp(
        "flashinfer::matmul_probe", "(Tensor a, Tensor b, Tensor(a!) out) -> ()",
        _MatmulProbeRunner(module, entry), matmul_probe_meta, ("out",),
    )


def _adapter_hash(recipe):
    from .core import _canonical_hash

    return _canonical_hash(recipe.output_adapter)


@lru_cache(maxsize=1)
def _acl_meta_api():
    # Runtime shared-library lookup only, never compiler/SDK discovery.
    import ctypes as C

    api = C.CDLL("libnnopbase.so", mode=C.RTLD_LOCAL)
    api.aclCreateTensor.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.c_int,
        C.POINTER(C.c_int64), C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
    api.aclCreateTensor.restype = C.c_void_p
    api.aclDestroyTensor.argtypes, api.aclDestroyTensor.restype = [C.c_void_p], C.c_int
    return api


@dataclass(frozen=True)
class _MatmulProbeRunner:
    module: object
    entry: object

    def __call__(self, a, b, out):
        import torch
        import torch_npu

        from flashinfer._meta.ascendc import matmul_probe_meta

        matmul_probe_meta(a, b, out)
        if a.device.type != torch._C._get_privateuse1_backend_name():
            raise ValueError("matmul_probe device execution requires PrivateUse1")
        if any(torch_npu.get_npu_format(t) != 2 for t in (a, b, out)):
            raise ValueError("matmul_probe requires NPU ND storage format")
        # Fixed contiguous extents make this a complete output-overlap check.
        pointers = [t.data_ptr() for t in (a, b, out)]
        if any(abs(pointers[2] - pointer) < 16 * 16 * 4 for pointer in pointers[:2]):
            raise ValueError("matmul_probe output must not overlap inputs")
        with torch.npu.device(a.device):
            stream = torch.npu.current_stream(a.device)
            for tensor in (a, b, out):
                tensor.record_stream(stream)
            if self.entry is not None:
                self.entry(stream.npu_stream, *pointers)
                stream.synchronize()
            else:
                self._launch_package(pointers, a.device, stream)

    def _launch_package(self, pointers, device, stream):
        import ctypes as C
        import torch

        api = _acl_meta_api()
        dims, strides = (C.c_int64 * 2)(16, 16), (C.c_int64 * 2)(16, 1)
        descriptors = []
        workspace = None
        try:
            for pointer in pointers:
                descriptor = api.aclCreateTensor(dims, 2, 0, strides, 0, 2, dims, 2, pointer)
                if not descriptor:
                    raise RuntimeError("aclCreateTensor failed")
                descriptors.append(descriptor)
            size, executor = self.module.get_workspace_size(*descriptors)
            if size:
                workspace = torch.empty(size, dtype=torch.uint8, device=device)
            self.module.launch(workspace.data_ptr() if workspace is not None else None,
                               size, executor, stream.npu_stream)
        finally:
            # This correctness probe is synchronous: keep descriptors/workspace
            # alive until CANN has consumed them, including exceptional paths.
            # It does not assert aclgraph capture/replay safety.
            try:
                stream.synchronize()
            finally:
                statuses = [api.aclDestroyTensor(d) for d in reversed(descriptors)]
                if any(statuses):
                    raise RuntimeError("aclDestroyTensor returned %s" % statuses)


__all__ = ["AscendCOp", "matmul_probe_op"]
