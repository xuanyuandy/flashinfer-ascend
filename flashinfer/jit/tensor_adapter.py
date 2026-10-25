"""Recipe-driven eager tensor adapter for the two-stage ACLNN package ABI.

This module owns tensor metadata/lifetimes, not package discovery or loading.
Tiling is produced by the package's host callback and retained by the ACLNN
executor; its device buffer is an implicit final kernel argument. Eager calls
are synchronous, so this adapter makes no graph-capture claim.
"""

from dataclasses import dataclass
from functools import lru_cache

from flashinfer.runtime import SchemaError


# aclDataType values from acl/acl_base.h; this is independent of operator names.
_ACL_DTYPES = {
    "float32": 0, "float16": 1, "int8": 2, "int32": 3, "uint8": 4,
    "int16": 6, "uint16": 7, "uint32": 8, "int64": 9, "uint64": 10,
    "float64": 11, "bool": 12, "bfloat16": 27,
}


@lru_cache(maxsize=1)
def _acl_tensor_api():
    import ctypes as C

    api = C.CDLL("libnnopbase.so", mode=C.RTLD_LOCAL)
    api.aclCreateTensor.argtypes = [
        C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64),
        C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p,
    ]
    api.aclCreateTensor.restype = C.c_void_p
    api.aclDestroyTensor.argtypes = [C.c_void_p]
    api.aclDestroyTensor.restype = C.c_int
    return api


def _dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _reject_output_overlap(parameters, tensors):
    for output_index, (parameter, output) in enumerate(zip(parameters, tensors)):
        if parameter.role != "output":
            continue
        out_lo, out_hi = output.data_ptr(), output.data_ptr() + output.numel() * output.element_size()
        for tensor_index, tensor in enumerate(tensors):
            if tensor_index == output_index:
                continue
            lo, hi = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
            if max(lo, out_lo) < min(hi, out_hi):
                raise ValueError("Ascend C output tensors must not overlap other arguments")


def package_tensor_runner(module, recipe, *, spec):
    """Validate a previously loaded package and return its generic adapter."""
    from .loading import JitLoadedModule
    from .op_package import OpPackageModule

    if (spec.kernel_id != recipe.kernel_id or spec.source != recipe.source
            or spec.target_soc != recipe.target_soc
            or spec.entry_points != (recipe.host_entry,)
            or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint):
        raise SchemaError("tensor adapter spec does not bind the recipe/entry")
    if isinstance(module, JitLoadedModule):
        if (module.receipt.spec_fingerprint != spec.fingerprint
                or tuple(symbol.name for symbol in module.receipt.symbols)
                != (recipe.host_entry,)):
            raise SchemaError("tensor adapter loaded receipt does not bind its spec")
        module = module.opaque_module
    if not isinstance(module, OpPackageModule) or module.recipe != recipe:
        raise SchemaError("tensor adapter requires the verified recipe's OpPackageModule")
    return TorchOpPackageRunner(module, recipe)


@dataclass(frozen=True)
class TorchOpPackageRunner:
    module: object
    recipe: object

    def __call__(self, *tensors):
        import ctypes as C
        import torch
        import torch_npu

        names = self.recipe.tensor_parameter_names
        if len(tensors) != len(names):
            raise SchemaError("tensor adapter expected parameters %s" % (names,))
        by_name = dict(zip(names, tensors))
        self.recipe.validate_tensor_metadata(
            shapes={name: tuple(t.shape) for name, t in by_name.items()},
            dtypes={name: _dtype_name(t) for name, t in by_name.items()},
        )
        device = tensors[0].device
        if device.type != torch._C._get_privateuse1_backend_name():
            raise ValueError("Ascend C tensor adapter requires NPU tensors")
        if any(t.device != device for t in tensors):
            raise ValueError("Ascend C tensors must share one device")
        if any(not t.is_contiguous() or torch_npu.get_npu_format(t) != 2 for t in tensors):
            raise ValueError("Ascend C tensor adapter requires contiguous ND tensors")
        _reject_output_overlap(self.recipe.tensor_parameters, tensors)

        api = _acl_tensor_api()
        handles = []
        metadata = []
        workspace = None
        with torch.npu.device(device):
            stream = torch.npu.current_stream(device)
            try:
                for tensor in tensors:
                    tensor.record_stream(stream)
                    dims = (C.c_int64 * tensor.ndim)(*tensor.shape)
                    strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
                    metadata.append((dims, strides))
                    dtype = _ACL_DTYPES.get(_dtype_name(tensor))
                    if dtype is None:
                        raise SchemaError("unsupported ACL tensor dtype: " + _dtype_name(tensor))
                    handle = api.aclCreateTensor(
                        dims, tensor.ndim, dtype, strides, 0, 2, dims, tensor.ndim,
                        tensor.data_ptr(),
                    )
                    if not handle:
                        raise RuntimeError("aclCreateTensor failed")
                    handles.append(handle)
                size, executor = self.module.get_workspace_size(*handles)
                if size:
                    workspace = torch.empty(size, device=device, dtype=torch.uint8)
                    workspace.record_stream(stream)
                self.module.launch(
                    workspace.data_ptr() if workspace is not None else None,
                    size, executor, stream.npu_stream,
                )
            finally:
                # All inputs, descriptors, workspace and host dimension arrays
                # remain live until queued device work finishes, even on error.
                try:
                    stream.synchronize()
                finally:
                    statuses = [api.aclDestroyTensor(handle) for handle in reversed(handles)]
                    if any(statuses):
                        raise RuntimeError("aclDestroyTensor returned %s" % statuses)


__all__ = ["TorchOpPackageRunner", "package_tensor_runner"]
