"""Recipe-driven eager tensor adapter for the two-stage ACLNN package ABI.

This module owns tensor metadata/lifetimes, not package discovery or loading.
Tiling is produced by the package's host callback and retained by the ACLNN
executor; its device buffer is an implicit final kernel argument. Eager calls
are synchronous, so this adapter makes no graph-capture claim.
"""

from dataclasses import dataclass, field
from threading import RLock

from flashinfer.runtime import SchemaError


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
    _native_runner: object = field(default=None, init=False, repr=False, compare=False)
    _native_lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __call__(self, *tensors):
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

        from flashinfer.aclnn import _native

        if self._native_runner is None:
            with self._native_lock:
                if self._native_runner is None:
                    object.__setattr__(self, "_native_runner", _native.prepare(self.module, self.recipe))
        return self._native_runner(tensors)


__all__ = ["TorchOpPackageRunner", "package_tensor_runner"]
