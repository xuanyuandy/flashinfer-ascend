"""Compiler hooks and pure declarations. No compiler SDK is imported here.

Backends describe one unit and invoke their compiler in a supplied staging
directory. The unified layer owns verification, cache/lock/publication, loading
and registration. Optional compiler packages are imported only by a build worker.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from typing import Optional, Protocol, Tuple, runtime_checkable

from flashinfer.runtime import Backend, KernelDescriptor, KernelRegistry, SchemaError
from flashinfer.version import __version__

from .core import JitSpec, JitSpecRegistry, _canonical_hash, jit_spec_registry
from .errors import JitCompilationError


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
    generator_version = "1"

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
            "generator_version": self.generator_version,
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
            generator_version=self.generator_version,
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
    """Compile a standalone Triton Python source in an isolated subprocess.

    Options are explicit ``signature=<JSON>``, ``constexprs=<JSON>`` and
    ``options=<JSON>`` objects. All three are part of the unchanged spec identity.
    Construction and declaration perform no I/O or optional imports.
    """

    backend_id = "triton"
    generator_version = "3"
    descriptor_backends = (Backend.TRITON, Backend.CUSTOM, Backend.EXTERNAL)

    def __init__(self, *, source_root: Optional[Path] = None,
                 python_executable: Optional[str] = None) -> None:
        self.source_root = Path(source_root) if source_root is not None else Path(__file__).parent.parent
        self.python_executable = python_executable or sys.executable

    def compile(self, spec: JitSpec, *, build_directory: Path) -> None:
        directory = Path(build_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "build.log"
        command = (self.python_executable, "-m", "flashinfer.jit._triton_compile",
                   str(directory / "request.json"))
        try:
            if not spec.is_compilation_unit or spec.generator_id != self.backend_id:
                raise ValueError("Triton compiler requires a triton compilation unit")
            config = {}
            for item in spec.compile_options:
                key, separator, value = item.partition("=")
                if not separator or key not in {"signature", "constexprs", "options"} or key in config:
                    raise ValueError("expected unique signature/constexprs/options=<JSON> compile options: %r" % item)
                config[key] = json.loads(value)
                if not isinstance(config[key], dict):
                    raise ValueError("%s must be a JSON object" % key)
            if not config.get("signature"):
                raise ValueError("Triton compilation requires an explicit signature=<JSON>")
            root = self.source_root.resolve()
            source_path = (root / spec.source.locator).resolve()
            if not source_path.is_relative_to(root):
                raise ValueError("source escapes source_root")
            payload = source_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != spec.source.digest:
                raise ValueError("SourceRef digest mismatch: %s" % source_path)
            snapshot = directory / "source.py"
            snapshot.write_bytes(payload)
            (directory / "request.json").write_text(json.dumps({
                "spec": spec.to_dict(), "config": config,
            }), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (
                str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH"))))
            env["TRITON_CACHE_DIR"] = str(directory / "triton-cache")
            env["TRITON_ALWAYS_COMPILE"] = "1"
            env["TRITON_STORE_BINARY_ONLY"] = "0"
            env["TRITON_KERNEL_OVERRIDE"] = "0"
            env["TRITON_KERNEL_DUMP"] = "0"
            with log_path.open("wb") as log:
                log.write(("command: %s\n" % repr(command)).encode())
                log.flush()
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        check=False)
            if result.returncode:
                tail = log_path.read_text(errors="replace")[-4000:]
                raise JitCompilationError(tail, kernel_id=spec.kernel_id,
                    spec_fingerprint=spec.fingerprint, command=command,
                    returncode=result.returncode, log_path=log_path)
        except JitCompilationError:
            raise
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                log.write("%s: %s\n" % (type(error).__name__, error))
            raise JitCompilationError(str(error), kernel_id=spec.kernel_id or spec.name,
                spec_fingerprint=spec.fingerprint, command=command, log_path=log_path) from error


class AscendCBackend(_SourceBackend):
    """Declare Ascend C units without locating or invoking the CANN compiler."""

    backend_id = "ascendc"
    descriptor_backends = (
        Backend.ASCENDC, Backend.ASCENDC_AOT, Backend.ASCENDC_JIT,
        Backend.CUSTOM, Backend.EXTERNAL,
    )

    def compile(self, spec: JitSpec, *, build_directory: Path) -> None:
        raise NotImplementedError(
            "ascendc compilation/loading remains unimplemented in T18B: bisheng/ccec "
            "can produce a device ELF, but a host wrapper and launch ABI have not "
            "been integrated or validated. No artifact was built or loaded."
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
