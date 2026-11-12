"""Ascend C authoring inputs. These fields do not belong to logical operators."""

from dataclasses import dataclass

from flashinfer.runtime import CapabilitySpec, SchemaError
from flashinfer.operators.implementation import ImplementationDefinition
from flashinfer.operators.author_api import Support


@dataclass(frozen=True)
class BuildConfig:
    device: str
    host: tuple
    op_type: str
    tiling: str
    includes: tuple = ("op_kernel", "op_host")
    cmake: object = None
    implementation: str = "native"
    dependencies: tuple = ()
    task_type: str = "aic_aiv"
    sync_policy: str = "cross_core_flags"
    workspace_bytes: object = "dynamic"
    version: int = 2
    definitions: tuple = ()
    compile_options: tuple = ()
    link_options: tuple = ()
    device_entry: object = None
    host_entry: object = None
    launch_abi: object = None
    package_vendor: object = None
    recipe_id: object = None
    tiling_mode: str = "host_dynamic"
    block_dim: int = 1


@dataclass(frozen=True)
class NativeDefinition:
    name: str
    kernel_id: str
    target_soc: str
    source_dir: str
    capability: CapabilitySpec
    build: BuildConfig
    tensors: tuple
    shapes: tuple
    dtype_signatures: tuple = ()
    attributes: tuple = ()
    workspace_bound: int = 0
    allow_jit_fallback: bool = False
    descriptor_metadata: object = None

    def __post_init__(self):
        if (
            not self.kernel_id.startswith("ascendc/")
            or not isinstance(self.capability, CapabilitySpec)
            or not isinstance(self.build, BuildConfig)
            or self.workspace_bound < 0
        ):
            raise SchemaError("invalid Ascend C implementation definition")

    @property
    def op(self):
        return self.capability.op


@dataclass(frozen=True)
class NativeBinding:
    """Actual native Tensor/attribute ABI; dimensions are in elements."""

    tensors: tuple
    dimensions: tuple = ()
    dtype_signatures: tuple = ()
    attributes: tuple = ()


class AscendCImplementation(ImplementationDefinition):
    """Author declaration that derives the native recipe and lazy execution binding."""

    def __init__(
        self,
        *,
        operator,
        support,
        build,
        binding,
        kernel_id=None,
        name="native",
        source_dir=None,
        runner_factory=None,
        workspace_bound=0,
        allow_jit_fallback=False,
        admission=None,
        **metadata
    ):
        from flashinfer.jit.recipe import ShapeConstraint

        if not isinstance(support, Support) or not isinstance(binding, NativeBinding):
            raise SchemaError(
                "Ascend C requires Support and NativeBinding declarations"
            )
        capability = support.capability(operator)
        if len(capability.soc_models) != 1:
            raise SchemaError(
                "one Ascend C compilation unit requires exactly one declared target SoC"
            )
        target = capability.soc_models[0]
        kernel_id = kernel_id or "ascendc/%s/%s/%s/%s" % (
            operator.domain,
            operator.name,
            name,
            target,
        )
        dimensions = binding.dimensions or tuple(
            ShapeConstraint(n, d.minimum, d.maximum, d.multiple_of)
            for n, d in support.dimensions.items()
        )
        native = NativeDefinition(
            name=build.recipe_id or operator.name,
            kernel_id=kernel_id,
            target_soc=target,
            source_dir=source_dir
            or "csrc/kernels/%s/%s" % (operator.domain, operator.name),
            capability=capability,
            build=build,
            tensors=binding.tensors,
            shapes=dimensions,
            dtype_signatures=binding.dtype_signatures,
            attributes=binding.attributes,
            workspace_bound=workspace_bound,
            allow_jit_fallback=allow_jit_fallback,
            descriptor_metadata=metadata or None,
        )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "runner_factory", runner_factory)
        object.__setattr__(self, "_admission", admission)
        super().__init__(
            kernel_id,
            operator.op,
            "ascendc",
            self._descriptor,
            origin="flashinfer.ascendc",
        )

    def _descriptor(self):
        from flashinfer.ascendc._runtime import bind_declaration

        return bind_declaration(self)

    def prepare_outputs(self, root):
        from pathlib import Path
        from flashinfer.ascendc.authoring import prepare_native

        directory = (
            Path(root)
            / "flashinfer/ascendc"
            / self.operator.domain
            / self.operator.name
        )
        return prepare_native(root, directory, self.native)
