"""Production Ascend C declarations and lazy AOT execution bindings.

Registration reads only packaged, frozen JSON metadata. Compiler discovery and
source resolution belong exclusively to an explicitly permitted JIT miss.
"""

from functools import lru_cache
import json
from pathlib import Path

from .runtime import SchemaError


MATMUL_BIAS_SILU_KERNEL_ID = "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950"
FLASH_ATTENTION_KERNEL_ID = "ascendc/t4/flash_attention/catlass/op_package/Ascend950"
_DATA = Path(__file__).parent / "data/kernels"


@lru_cache(maxsize=1)
def _contracts():
    from .jit.core import JitSpec
    from .jit.recipe import BuildRecipe
    from .runtime.manifest import load_kernel_manifest

    descriptors = load_kernel_manifest(_DATA / "registry.json")
    payload = json.loads((_DATA / "ascendc_specs.json").read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "kernels"} or payload["schema_version"] != 1:
        raise SchemaError("unsupported frozen Ascend C declaration schema")
    frozen = {}
    for record in payload["kernels"]:
        if set(record) != {"kernel_id", "recipe", "spec"} or record["kernel_id"] in frozen:
            raise SchemaError("invalid or duplicate frozen Ascend C declaration")
        frozen[record["kernel_id"]] = record
    if set(frozen) != {descriptor.kernel_id for descriptor in descriptors}:
        raise SchemaError("registry and frozen Ascend C declarations disagree")
    contracts = {}
    for descriptor in descriptors:
        record = frozen[descriptor.kernel_id]
        recipe = BuildRecipe.from_dict(record["recipe"])
        spec = JitSpec.from_dict(record["spec"])
        recipe.validate_descriptor(descriptor)
        spec.validate_descriptor(descriptor)
        if dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint:
            raise SchemaError("frozen Ascend C spec does not bind its recipe")
        contracts[descriptor.kernel_id] = (descriptor, recipe, spec)
    return contracts


def load_contract(kernel_id=MATMUL_BIAS_SILU_KERNEL_ID):
    """Read trusted installed metadata without touching sources or a compiler."""
    try:
        return _contracts()[kernel_id]
    except KeyError as error:
        raise SchemaError("no frozen Ascend C declaration for " + kernel_id) from error


def _dependency_roots(recipe, source_root):
    """Resolve explicit build dependencies only after JIT/build is authorized."""
    import os

    if not any(dependency.name == "catlass" for dependency in recipe.dependencies):
        return {}
    location = Path(os.environ.get("FLASHINFER_CATLASS_ROOT", Path(source_root) / "3rdparty/catlass"))
    dependency = next(item for item in recipe.dependencies if item.name == "catlass")
    if not (location / dependency.include_dir).is_dir():
        raise SchemaError(
            "catlass dependency headers are missing at %s; set FLASHINFER_CATLASS_ROOT "
            "to the pinned catlass checkout (headers are not bundled in the wheel)" % location
        )
    return {"catlass": location}


def _jit_backend(recipe, spec):
    from .aot.build import _source_root
    from .jit import AscendCBackend, AscendCToolchain, resolve_build_recipe

    toolchain = AscendCToolchain.discover()
    if toolchain.fingerprint != spec.environment_fingerprint:
        freeze_tool = ("build_flash_attention.py" if recipe.kernel_id == FLASH_ATTENTION_KERNEL_ID
                       else "build_matmul_bias_silu.py")
        raise SchemaError(
            "installed Ascend C declaration requires its frozen build environment; "
            "current toolchain fingerprint does not match; rebuild/freeze the release "
            "with tools/%s --freeze" % freeze_tool
        )
    source_root = _source_root()
    resolved = resolve_build_recipe(
        recipe, source_root=source_root,
        toolchain_fingerprint=toolchain.fingerprint,
        dependency_roots=_dependency_roots(recipe, source_root),
    )
    if dict(spec.input_fingerprints).get("resolved_recipe") != resolved.fingerprint:
        raise SchemaError("installed Ascend C declaration does not match resolved build sources")
    return AscendCBackend(recipes=(resolved,), toolchain=toolchain)


def ascendc_descriptors():
    """Bind packaged production entries for the ordinary global registry."""
    from .jit.runtime import bind_aot_kernel
    from .jit.tensor_adapter import package_tensor_runner

    return tuple(
        bind_aot_kernel(
            descriptor, spec=spec,
            runner_factory=lambda module, loaded_spec, recipe=recipe: package_tensor_runner(
                module, recipe, spec=loaded_spec,
            ),
            jit_backend_factory=lambda recipe=recipe, spec=spec: _jit_backend(recipe, spec),
        )
        for descriptor, recipe, spec in _contracts().values()
    )


def tensor_contract_specialization(kernel_id, tensors):
    """Expose recipe shape/type validation as a normal dispatch constraint.

    The diagnostic becomes a signature specialization, so the existing
    dispatcher rejects unsupported metadata before invoking any lazy runner.
    """
    _, recipe, _ = load_contract(kernel_id)
    parameters = recipe.tensor_parameter_names
    if len(tensors) != len(parameters):
        return "recipe requires tensor parameters %s" % (parameters,)
    try:
        recipe.validate_tensor_metadata(
            shapes={name: tuple(tensor.shape) for name, tensor in zip(parameters, tensors)},
            dtypes={name: str(tensor.dtype).removeprefix("torch.")
                    for name, tensor in zip(parameters, tensors)},
        )
        devices = {str(tensor.device) for tensor in tensors}
        if len(devices) != 1:
            return "all tensors must share one device"
        if any(not tensor.is_contiguous() for tensor in tensors):
            return "all tensors must be contiguous row-major"
    except (SchemaError, ValueError, TypeError) as error:
        return str(error)
    return "supported"


__all__ = ["MATMUL_BIAS_SILU_KERNEL_ID", "FLASH_ATTENTION_KERNEL_ID", "load_contract", "ascendc_descriptors"]
