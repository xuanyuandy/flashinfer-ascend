"""Typed author declarations; manifests, defaults and content hashes are derived."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from flashinfer.runtime import (
    Backend,
    CapabilitySpec,
    CompileMode,
    KernelDescriptor,
    SchemaError,
    ShapeRange,
    SourceRef,
)
from .definition import OperatorDefinition
from .implementation import ImplementationDefinition


@dataclass(frozen=True)
class DimRange:
    """Inclusive logical dimension bounds in elements, with required alignment."""

    minimum: int
    maximum: int
    multiple_of: int = 1

    def __post_init__(self):
        if (
            any(
                type(v) is not int
                for v in (self.minimum, self.maximum, self.multiple_of)
            )
            or self.minimum < 1
            or self.maximum < self.minimum
            or self.multiple_of < 1
        ):
            raise SchemaError(
                "dimension bounds require positive integers and minimum <= maximum"
            )


class Support:
    """Input support rules, with named dimensions bound to the logical definition.

    soc/dtype/layout abbreviate soc_models/dtypes/physical_layouts. Advanced
    coupled rules retain the existing CapabilityCase semantics unchanged.
    """

    def __init__(self, *, dimensions=None, contiguous=False, **rules):
        for alias, target in (
            ("soc", "soc_models"),
            ("dtype", "dtypes"),
            ("layout", "physical_layouts"),
        ):
            if alias in rules:
                if target in rules:
                    raise SchemaError(
                        "support declares both %s and %s" % (alias, target)
                    )
                value = rules.pop(alias)
                rules[target] = (value,) if isinstance(value, str) else tuple(value)
        unknown = set(rules) - (set(CapabilitySpec.__dataclass_fields__) - {"op"})
        if unknown:
            raise SchemaError("unknown support fields: " + ", ".join(sorted(unknown)))
        self.rules = dict(rules)
        self.dimensions = dict(dimensions or {})
        self.contiguous = contiguous
        if type(contiguous) is not bool:
            raise SchemaError("contiguous must be bool")
        if any(not isinstance(v, DimRange) for v in self.dimensions.values()):
            raise SchemaError("named dimensions require DimRange values")

    def capability(self, operator):
        rules = dict(self.rules)
        if self.dimensions:
            names = {
                n
                for shape in operator.tensor_shapes
                for n in shape
                if isinstance(n, str)
            }
            if set(self.dimensions) - names:
                raise SchemaError(
                    "support dimension is absent from the logical Tensor shapes"
                )
            ranges = list(rules.get("shape_ranges", ()))
            for index, shape in enumerate(operator.tensor_shapes):
                for axis, name in enumerate(shape):
                    if name in self.dimensions:
                        limit = self.dimensions[name]
                        ranges.append(
                            ShapeRange(name, index, axis, limit.minimum, limit.maximum)
                        )
            rules["shape_ranges"] = tuple(ranges)
        return CapabilitySpec(operator.op, **rules)

    def admission(self, operator, signature):
        if self.contiguous and "0" in dict(signature.specializations).get(
            "tensor_contiguity", ""
        ):
            return ("implementation requires contiguous Tensors",)
        for index, axes in enumerate(operator.tensor_shapes):
            if index >= len(signature.shapes):
                return ("implementation requires its complete Tensor signature",)
            for axis, name in enumerate(axes):
                if name in self.dimensions:
                    multiple = self.dimensions[name].multiple_of
                    if (
                        axis >= len(signature.shapes[index])
                        or signature.shapes[index][axis] % multiple
                    ):
                        return (
                            "implementation requires %s multiple_of=%d"
                            % (name, multiple),
                        )
        return ()


@dataclass(frozen=True)
class PythonKernel:
    """Python device source relative to its implementation directory."""

    file: str = "kernels.py"

    def source_ref(self, directory, package):
        path = directory / self.file
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
        ):
            raise SchemaError(
                "kernel source must be a regular implementation-local file"
            )
        return SourceRef(
            "triton",
            path.relative_to(package).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )


class PythonImplementation(ImplementationDefinition):
    """Backend author API backed by the existing immutable declaration protocol."""

    BACKEND = Backend.REFERENCE

    def __init__(
        self,
        *,
        operator,
        support=None,
        runner=None,
        runner_factory=None,
        kernel_id=None,
        name="default",
        source=None,
        scope="global",
        origin=None,
        admission=None,
        **metadata
    ):
        if not isinstance(operator, OperatorDefinition):
            raise SchemaError("implementation requires a logical OperatorDefinition")
        if (runner is None) == (runner_factory is None):
            raise SchemaError("provide exactly one of runner or runner_factory")
        if not callable(runner if runner is not None else runner_factory):
            raise SchemaError("runner or runner_factory must be callable")
        support = support or Support()
        if not isinstance(support, Support):
            raise SchemaError("implementation support must be Support")
        backend = self.BACKEND
        kernel_id = kernel_id or "%s/%s/%s/%s" % (
            backend.value,
            operator.domain,
            operator.name,
            name,
        )
        package = Path(__file__).resolve().parents[1]
        for key, value in dict(
            operator=operator,
            support=support,
            source=source,
            directory=package / backend.value / operator.domain / operator.name,
            _runner=runner,
            _runner_factory=runner_factory,
            _metadata=dict(metadata),
            _admission=admission,
        ).items():
            object.__setattr__(self, key, value)
        if origin is None:
            origin = (
                "flashinfer.runtime.reference_kernels"
                if backend == Backend.REFERENCE
                else "flashinfer.%s.%s" % (backend.value, operator.domain)
            )
        super().__init__(
            kernel_id,
            operator.op,
            backend,
            self._descriptor,
            scope=scope,
            origin=origin,
        )

    def _descriptor(self):
        metadata = dict(self._metadata)
        if self.source is not None:
            metadata["source"] = self.source.source_ref(
                self.directory, Path(__file__).resolve().parents[1]
            )
        if self.BACKEND == Backend.TRITON:
            metadata.setdefault("compile_mode", CompileMode.JIT)
        value = KernelDescriptor(
            self.kernel_id,
            op=self.op,
            backend=self.backend,
            capability=self.support.capability(self.operator),
            **metadata
        )
        if self.operator.schema and value.capability_binding is None:
            from .capability import direct_operator_binding

            value = replace(value, capability_binding=direct_operator_binding(
                self.operator, value.capability, value.backend, value.kernel_id))
        runner = (
            self._runner if self._runner is not None else self._runner_factory(value)
        )

        def admission(signature):
            reasons = self.support.admission(self.operator, signature)
            if self._admission is not None:
                reasons += tuple(self._admission(signature))
            return reasons

        return replace(value, runner=runner, admission=admission)


class CANNImplementation(PythonImplementation):
    BACKEND = Backend.CANN


class TritonImplementation(PythonImplementation):
    BACKEND = Backend.TRITON


class ReferenceImplementation(PythonImplementation):
    BACKEND = Backend.REFERENCE
