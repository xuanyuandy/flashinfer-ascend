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
        recipe,
        source_root=source_root,
        toolchain_fingerprint=toolchain.fingerprint,
        dependency_roots=_dependency_roots(recipe, source_root),
    )
    if dict(spec.input_fingerprints).get("resolved_recipe") != resolved.fingerprint:
        raise SchemaError(
            "installed Ascend C declaration does not match resolved build sources"
        )
    return AscendCBackend(recipes=(resolved,), toolchain=toolchain)


def tensor_contract_specialization(kernel_id, tensors):
    """Expose recipe shape/type validation as a normal dispatch constraint.

    The diagnostic becomes a signature specialization, so the existing
    dispatcher rejects unsupported metadata before invoking any lazy runner.
    """
    from ._catalog import source_contract

    prepared = source_contract(kernel_id)
    recipe = prepared["recipe"] if prepared is not None else load_contract(kernel_id)[1]
    parameters = recipe.tensor_parameter_names
    if len(tensors) != len(parameters):
        return "recipe requires tensor parameters %s" % (parameters,)
    try:
        recipe.validate_tensor_metadata(
            shapes={
                name: tuple(tensor.shape) for name, tensor in zip(parameters, tensors)
            },
            dtypes={
                name: str(tensor.dtype).removeprefix("torch.")
                for name, tensor in zip(parameters, tensors)
            },
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
            shapes={
                name: None if dtype == "none" else shape
                for name, shape, dtype in zip(names, signature.shapes, signature.dtypes)
            },
            dtypes={
                name: None if dtype == "none" else dtype
                for name, dtype in zip(names, signature.dtypes)
            },
        )
    except (ValueError, TypeError) as error:
        return (str(error),)
    for parameter, dtype, layout in zip(
        recipe.tensor_parameters, signature.dtypes, signature.physical_layouts
    ):
        if dtype != "none" and layout and layout not in parameter.physical_layouts:
            return (
                "Tensor format violates declared physical_layouts: " + parameter.name,
            )
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
    make_runner = runner_factory or (
        lambda module, loaded, recipe: package_tensor_runner(
            module, recipe, spec=loaded
        )
    )
    bound = bind_aot_kernel(
        descriptor,
        spec=spec,
        runner_factory=lambda module, loaded: make_runner(module, loaded, recipe),
        jit_backend_factory=(
            (lambda: ascendc._jit_backend(recipe, spec))
            if recipe.allow_jit_fallback
            else None
        ),
        package_peers=peers,
    )
    return replace(
        bound, admission=lambda signature: native_admission(recipe, signature)
    )


def bind_declaration(declaration):
    """Source metadata is live; the frozen build contract is required on use."""
    from dataclasses import replace
    from functools import cached_property
    from threading import RLock
    from flashinfer._resources import source_root, data_root
    from flashinfer.jit.runtime import AotKernelRunner
    from . import _catalog

    root = source_root()
    if _catalog._DATA != data_root() / "kernels":
        bound = bind_kernel(declaration.kernel_id, declaration.runner_factory)
        recipe = load_contract(declaration.kernel_id)[1]
    else:
        if root is not None:
            prepared = declaration.prepare_outputs(root)
            descriptor, recipe = prepared["descriptor"], prepared["recipe"]
        else:
            descriptor, recipe = _catalog.declarations()[declaration.kernel_id]

        class SourceRunner(AotKernelRunner):
            def __init__(self):
                self.descriptor = descriptor
                self._lock = RLock()
                self._runner = self._module = self._provenance = None
                self.index = self.policy = self.store = None
                self.jit_backend_factory = (
                    (lambda: self._frozen_runner.jit_backend_factory())
                    if recipe.allow_jit_fallback
                    else None
                )

            @cached_property
            def _frozen_runner(self):
                return bind_kernel(
                    declaration.kernel_id, declaration.runner_factory
                ).runner

            @property
            def spec(self):
                return self._frozen_runner.spec

            @property
            def runner_factory(self):
                return self._frozen_runner.runner_factory

            @property
            def package_peers(self):
                return self._frozen_runner.package_peers

        bound = replace(descriptor, runner=SourceRunner())
    admit = declaration._admission or native_admission
    return replace(bound, admission=lambda signature: admit(recipe, signature))
