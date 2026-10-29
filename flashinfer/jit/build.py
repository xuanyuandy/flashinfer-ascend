"""Explicit lazy build -> verify -> publish -> load orchestration."""

import hashlib
import json
import os
import platform
import traceback
from pathlib import Path
from threading import RLock
from typing import Optional

from flashinfer.runtime import ArtifactFormat, ArtifactKind, ArtifactRef, SchemaError

from ._elf import elf_symbols
from .artifacts import verify_jit_cache_record_payload
from .backends import JitBackend
from .cache import DiskJitBuildStore, JitCacheIndex, MissingJITCacheError, jit_cache_index
from .core import JitSpec
from .env import JitCompilationPolicy
from .errors import JitBuildError, JitCompilationError, JitSymbolError
from .loading import DefaultJitModuleLoader, JitModuleLoader, _triton_files


_loaders = {}
_loaders_lock = RLock()


def _verify_output(spec: JitSpec, directory: Path) -> ArtifactRef:
    path = (directory / spec.output_path).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()):
        raise SchemaError("compiled output escapes staging root")
    payload = path.read_bytes()
    adapter = dict(spec.input_fingerprints).get("output_adapter")
    if adapter is not None:
        from .op_package import OP_PACKAGE_ADAPTER_HASH, verify_op_package

        if spec.generator_id != "ascendc" or adapter != OP_PACKAGE_ADAPTER_HASH:
            raise SchemaError("unknown output adapter")
        verify_op_package(directory, spec)
        format = ArtifactFormat.ASCENDC_OP_PACKAGE
        symbols = spec.entry_points  # verified against the package's host ELF
    elif spec.source.language == "triton":
        format = ArtifactFormat.TRITON_NPUBIN
        from ._triton_abi import triton_binary_mode, verify_triton_symbols
        config = {key: json.loads(value) for key, _, value in (
            option.partition("=") for option in spec.compile_options) if key == "binary"}
        binary_mode = triton_binary_mode(config)
        data = json.loads((directory / "triton.json").read_text())
        if data.get("binary_mode", "single") != binary_mode:
            raise SchemaError("Triton binary mode does not match the declared spec")
        try:
            verify_triton_symbols(payload, spec.entry_points[0], binary_mode)
        except SchemaError as error:
            raise JitSymbolError(str(error), kernel_id=spec.kernel_id,
                spec_fingerprint=spec.fingerprint, log_path=directory / "build.log") from error
        # The logical entry binds either one entry or the complete mixed pair.
        symbols = spec.entry_points
    else:
        format = ArtifactFormat.SHARED_LIBRARY
        machines = {"x86_64": 62, "aarch64": 183}
        symbols = elf_symbols(payload, machine=machines[platform.machine()], elf_type=3, dynamic_only=True)
    if spec.entry_points[0] not in symbols:
        raise JitSymbolError("compiled ELF has no defined entry symbol %r" % spec.entry_points[0],
            kernel_id=spec.kernel_id, spec_fingerprint=spec.fingerprint,
            log_path=directory / "build.log")
    artifact = ArtifactRef(ArtifactKind.FILE, format, spec.output_path,
        hashlib.sha256(payload).hexdigest(), spec.target_soc, spec.fingerprint, len(payload),
        entry_point=spec.entry_points[0])
    if format == ArtifactFormat.TRITON_NPUBIN:
        _triton_files(directory, artifact, spec.fingerprint)
    artifact.verify_file(directory)
    return artifact


def _build_record(
    spec: JitSpec, *, backend: JitBackend, store: DiskJitBuildStore,
    policy: Optional[JitCompilationPolicy] = None,
):
    """The single build/verify/publish path, also used by release-time AOT."""
    if not isinstance(spec, JitSpec) or not spec.is_compilation_unit:
        raise SchemaError("build_and_load requires a JIT compilation unit")
    if not isinstance(backend, JitBackend) or backend.backend_id != spec.generator_id:
        raise SchemaError("compiler backend does not match JIT spec generator")
    if not isinstance(store, DiskJitBuildStore):
        raise TypeError("build_and_load requires DiskJitBuildStore")
    if policy is None:
        policy = (JitCompilationPolicy.DISABLED if os.environ.get("FLASHINFER_NPU_JIT_DISABLE", "0") == "1"
                  else JitCompilationPolicy.ENABLED)
    policy = JitCompilationPolicy(policy)
    record = store.lookup(spec)
    if record is None:
        if policy != JitCompilationPolicy.ENABLED:
            raise MissingJITCacheError("JIT disk cache miss for %s (%s); compilation policy=%s"
                                       % (spec.kernel_id, spec.fingerprint, policy.value))
        with store.lock(spec):
            record = store.lookup(spec)  # Required second check under the stable lock.
            if record is None:
                with store.staging(spec) as directory:
                    try:
                        backend.compile(spec, build_directory=directory)
                        artifact = _verify_output(spec, directory)
                        record = store.publish(spec, directory, artifact)
                    except JitBuildError as error:
                        log_path = directory / "build.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(traceback.format_exc())
                        error.log_path = log_path
                        raise
                    except NotImplementedError:
                        raise
                    except Exception as error:
                        log_path = directory / "build.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(traceback.format_exc())
                        raise JitCompilationError(str(error), kernel_id=spec.kernel_id,
                            spec_fingerprint=spec.fingerprint, log_path=log_path) from error
    record.validate_spec(spec)
    return record


def build_and_load(
    spec: JitSpec, *, backend: JitBackend, store: Optional[DiskJitBuildStore] = None,
    loader: Optional[JitModuleLoader] = None, cache: Optional[JitCacheIndex] = None,
    policy: Optional[JitCompilationPolicy] = None,
):
    """Build on a disk miss under the shared store lock, then load verified bytes.

    Neither declaration, status queries nor dispatch invokes this function.
    """
    if cache is not None and not isinstance(cache, JitCacheIndex):
        raise TypeError("cache must be JitCacheIndex")
    if loader is not None and not isinstance(loader, JitModuleLoader):
        raise TypeError("loader must implement JitModuleLoader")
    if store is None:
        store = DiskJitBuildStore()
    if cache is None:
        cache = jit_cache_index
    record = _build_record(spec, backend=backend, store=store, policy=policy)
    path = store.artifact_path(record)
    verification = verify_jit_cache_record_payload(record, path.read_bytes(), verifier_id="flashinfer.disk")
    # Compilation is recorded independently of successful loading. A symbol or
    # device-load failure must not claim that verified compiled bytes are absent.
    cache.publish(record, artifact_path=path)
    if loader is None:
        with _loaders_lock:
            key = str(store.root)
            loader = _loaders.get(key)
            if loader is None:
                loader = DefaultJitModuleLoader(store, max_entries=int(os.environ.get(
                    "FLASHINFER_NPU_JIT_CACHE_SIZE", "1024")))
                _loaders[key] = loader
    loaded = loader.load(record, verification, spec.entry_points)
    loaded.receipt.validate(record, verification, spec.entry_points)
    return loaded


__all__ = ["build_and_load"]
