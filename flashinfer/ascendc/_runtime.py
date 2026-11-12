"""Shared lazy Ascend C execution binding, independent of operator domains."""

from flashinfer.runtime import SchemaError
from ._catalog import _contracts, load_contract

def _dependency_roots(recipe, source_root):
    """Compatibility entry into the shared AOT/JIT dependency resolver."""
    from flashinfer.jit.dependencies import dependency_roots

    return dependency_roots(recipe, source_root)

def _jit_backend(recipe, spec):
    from flashinfer.aot.build import _source_root
    from flashinfer.jit import AscendCBackend, AscendCToolchain, resolve_build_recipe

    toolchain = AscendCToolchain.discover()
    if toolchain.fingerprint != spec.environment_fingerprint:
        raise SchemaError(
            "installed Ascend C declaration requires its frozen build environment; "
            "current toolchain fingerprint does not match; rebuild/freeze the release "
            "with python -m flashinfer.aot --freeze --kernel-id %s" % recipe.kernel_id
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

def native_admission(recipe, signature):
    names = recipe.tensor_parameter_names
    if len(signature.shapes) != len(names) or len(signature.dtypes) != len(names):
        return ("native implementation requires its complete Tensor signature",)
    try:
        recipe.validate_tensor_metadata(
            shapes={name: None if dtype == "none" else shape
                    for name, shape, dtype in zip(names, signature.shapes, signature.dtypes)},
            dtypes={name: None if dtype == "none" else dtype for name, dtype in zip(names, signature.dtypes)})
    except (ValueError, TypeError) as error:
        return (str(error),)
    for parameter, dtype, layout in zip(recipe.tensor_parameters, signature.dtypes, signature.physical_layouts):
        if dtype != "none" and layout and layout not in parameter.physical_layouts:
            return ("Tensor format violates declared physical_layouts: " + parameter.name,)
    if "0" in dict(signature.specializations).get("tensor_contiguity", ""):
        return ("Ascend C implementation requires contiguous Tensors",)
    return ()


def bind_kernel(kernel_id, runner_factory=None):
    from dataclasses import replace
    from flashinfer.jit.runtime import bind_aot_kernel
    from flashinfer.jit.tensor_adapter import package_tensor_runner
    from flashinfer import ascendc

    descriptor, recipe, spec = load_contract(kernel_id)
    peers = tuple((d, s) for d, _, s in _contracts().values())
    make_runner = runner_factory or (lambda module, loaded, recipe: package_tensor_runner(module, recipe, spec=loaded))
    bound = bind_aot_kernel(
        descriptor, spec=spec,
        runner_factory=lambda module, loaded: make_runner(module, loaded, recipe),
        jit_backend_factory=(lambda: ascendc._jit_backend(recipe, spec)) if recipe.allow_jit_fallback else None,
        package_peers=peers,
    )
    return replace(bound, admission=lambda signature: native_admission(recipe, signature))
