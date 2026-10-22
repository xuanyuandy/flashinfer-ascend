"""Compiler hooks and pure declarations. No compiler SDK is imported here.

Backends describe one unit and invoke their compiler in a supplied staging
directory. The unified layer owns verification, cache/lock/publication, loading
and registration. T18A implements only declarations; compiler hooks fail
explicitly until T18B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Tuple, runtime_checkable

from flashinfer.runtime import Backend, KernelDescriptor, KernelRegistry, SchemaError
from flashinfer.version import __version__

from .core import JitSpec, JitSpecRegistry, _canonical_hash, jit_spec_registry


@runtime_checkable
class JitBackend(Protocol):
    """The two backend hooks; neither owns cache, loading or registration.

    ``compile`` must write the declared output relative to ``build_directory``
    or raise. Returning normally is not a verified build receipt: T18B's
    unified layer must verify output bytes before constructing an ArtifactRef.
    Native output formats and auxiliary files remain T18B integration work.
    """

    backend_id: str

    def declare(
        self,
        descriptor: KernelDescriptor,
        *,
        target_soc: str,
        output_path: str,
        entry_point: str,
        environment_fingerprint: str,
        specialization_fingerprint: str,
        compile_options: Tuple[str, ...] = (),
    ) -> JitSpec:
        """Return a pure declaration using descriptor.source, without I/O."""
        raise NotImplementedError("JIT declaration hook must be implemented by the backend")

    def compile(self, spec: JitSpec, *, build_directory: Path) -> None:
        """Invoke only the compiler, using a unified-layer staging directory."""
        raise NotImplementedError("JIT compiler hook awaits T18B")


class _SourceBackend:
    backend_id: str
    descriptor_backends: Tuple[Backend, ...]

    def declare(
        self,
        descriptor: KernelDescriptor,
        *,
        target_soc: str,
        output_path: str,
        entry_point: str,
        environment_fingerprint: str,
        specialization_fingerprint: str,
        compile_options: Tuple[str, ...] = (),
    ) -> JitSpec:
        if not isinstance(descriptor, KernelDescriptor):
            raise TypeError("descriptor must be KernelDescriptor")
        if descriptor.source is None:
            raise SchemaError("kernel %r has no SourceRef declaration" % descriptor.kernel_id)
        if (descriptor.source.language != self.backend_id
                or descriptor.backend not in self.descriptor_backends):
            raise SchemaError("%s backend requires matching source language and kernel backend"
                              % self.backend_id)
        options = _compile_options(compile_options)
        identity = {
            "backend": self.backend_id,
            "generator_version": "1",
            "kernel": descriptor.fingerprint,
            "target_soc": target_soc,
            "output_path": output_path,
            "entry_point": entry_point,
            "environment": environment_fingerprint,
            "specialization": specialization_fingerprint,
            "compile_options": options,
            "package_version": __version__,
        }
        spec = JitSpec(
            name="kernel." + _canonical_hash(identity),
            domain="kernel",
            generator_id=self.backend_id,
            generator_version="1",
            target_soc=target_soc,
            environment_fingerprint=environment_fingerprint,
            specialization_fingerprint=specialization_fingerprint,
            input_fingerprints=(
                ("kernel", descriptor.fingerprint),
                ("package", _canonical_hash({"version": __version__})),
            ),
            compile_options=options,
            entry_points=(entry_point,),
            kernel_id=descriptor.kernel_id,
            source=descriptor.source,
            output_path=output_path,
        )
        spec.validate_descriptor(descriptor)
        return spec

    def compile(self, spec: JitSpec, *, build_directory: Path) -> None:
        raise NotImplementedError(
            "%s compiler invocation is not implemented; awaits T18B. "
            "T18A only declares compilation units; no artifact was built or loaded."
            % self.backend_id
        )


class TritonBackend(_SourceBackend):
    """Declare Triton units without importing triton or its decorated source."""

    backend_id = "triton"
    descriptor_backends = (Backend.TRITON, Backend.CUSTOM, Backend.EXTERNAL)


class AscendCBackend(_SourceBackend):
    """Declare Ascend C units without locating or invoking the CANN compiler."""

    backend_id = "ascendc"
    descriptor_backends = (
        Backend.ASCENDC, Backend.ASCENDC_AOT, Backend.ASCENDC_JIT,
        Backend.CUSTOM, Backend.EXTERNAL,
    )


def _compile_options(options: Tuple[str, ...]) -> Tuple[str, ...]:
    if (not isinstance(options, (tuple, list))
            or any(not isinstance(item, str) for item in options)):
        raise SchemaError("JIT compile options must be a list or tuple of strings")
    return tuple(options)


def declare_kernel(
    kernel_id: str,
    *,
    backend: JitBackend,
    output_path: str,
    entry_point: str,
    environment_fingerprint: str,
    specialization_fingerprint: str,
    target_soc: Optional[str] = None,
    compile_options: Tuple[str, ...] = (),
    registry: Optional[KernelRegistry] = None,
    spec_registry: Optional[JitSpecRegistry] = None,
) -> JitSpec:
    """Generate, validate and register a declaration through the same entry.

    The KernelRegistry is read only. No backend is inferred from dispatch
    policy: the caller explicitly supplies the source-language implementation.
    Environment and specialization identities must also be supplied explicitly.
    """
    if not isinstance(backend, JitBackend):
        raise TypeError("backend must implement JitBackend.declare and JitBackend.compile")
    if registry is None:
        from flashinfer.runtime.registry import get_global_registry

        registry = get_global_registry()
    if spec_registry is None:
        spec_registry = jit_spec_registry
    descriptor = registry.get(kernel_id)
    recorded_target = (descriptor.artifact.target_soc if descriptor.artifact is not None
                       else descriptor.target_soc)
    target = target_soc if target_soc is not None else recorded_target
    if target is None:
        raise SchemaError("kernel %r requires an explicit target_soc" % kernel_id)
    options = _compile_options(compile_options)
    spec = backend.declare(
        descriptor, target_soc=target, output_path=output_path, entry_point=entry_point,
        environment_fingerprint=environment_fingerprint,
        specialization_fingerprint=specialization_fingerprint, compile_options=options,
    )
    if not isinstance(spec, JitSpec):
        raise TypeError("JitBackend.declare must return JitSpec")
    spec.validate_descriptor(descriptor)
    if (spec.generator_id != backend.backend_id or spec.target_soc != target
            or spec.output_path != output_path or spec.entry_points != (entry_point,)
            or spec.compile_options != options
            or spec.environment_fingerprint != environment_fingerprint
            or spec.specialization_fingerprint != specialization_fingerprint):
        raise SchemaError("backend declaration does not match the requested compilation unit")
    return spec_registry.register(spec)


__all__ = ["JitBackend", "TritonBackend", "AscendCBackend", "declare_kernel"]
