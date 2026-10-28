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
    """Prepare the v2 FP32 probe through the common verified Tensor adapter.

    Public declarations accept only the explicit three-Tensor ACLNN contract.
    Raw host launch and v1 decoding remain internal compatibility facilities.
    """
    from flashinfer._meta.ascendc import matmul_probe_meta
    from flashinfer.runtime import SchemaError
    from .recipe import BuildRecipe
    from .tensor_adapter import package_tensor_runner

    if (not isinstance(recipe, BuildRecipe) or recipe.schema_version != 2
            or recipe.recipe_id != "matmul_probe"
            or recipe.launch_abi != "aclnn.matmul_probe.v2"):
        raise SchemaError("matmul_probe_op requires the public v2 ACLNN probe recipe")
    if spec is None:
        raise SchemaError("v2 matmul_probe_op requires its verified spec")
    return AscendCOp(
        "flashinfer::matmul_probe", "(Tensor a, Tensor b, Tensor(a!) out) -> ()",
        package_tensor_runner(module, recipe, spec=spec), matmul_probe_meta, ("out",),
    )


__all__ = ["AscendCOp", "matmul_probe_op"]
