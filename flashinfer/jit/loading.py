"""JIT loading protocol, receipts, and lazy Triton/host library loading."""

from __future__ import annotations

import hashlib
import json
import re
import os
import sys
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from threading import RLock
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Tuple, runtime_checkable

from flashinfer.runtime import ArtifactFormat, ArtifactRef, SchemaError

from .artifacts import JitArtifactVerification
from .cache import DiskJitBuildStore, JitCacheRecord
from .errors import JitLoadError, JitSymbolError
from ._elf import elf_symbols


JIT_MODULE_LOAD_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise SchemaError("JIT loaded module %s must be lowercase SHA-256" % name)


@dataclass(frozen=True)
class JitResolvedSymbol:
    """Opaque symbol identity; never a callable or process address."""

    name: str
    symbol_token: str
    schema_version: int = JIT_MODULE_LOAD_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_MODULE_LOAD_VERSION:
            raise SchemaError("unsupported JIT resolved symbol version")
        if not _SYMBOL.fullmatch(str(self.name)):
            raise SchemaError("invalid JIT resolved symbol name")
        if not str(self.symbol_token) or any(
            item in str(self.symbol_token) for item in ("\x00", "\n", "\r")
        ):
            raise SchemaError("invalid JIT resolved symbol token")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "symbol_token", str(self.symbol_token))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "symbol_token": self.symbol_token,
        }


@dataclass(frozen=True)
class JitModuleLoadReceipt:
    """Identity of one loaded artifact and its exact exported symbols."""

    spec_name: str
    spec_fingerprint: str
    cache_record_fingerprint: str
    artifact_verification_fingerprint: str
    artifact_fingerprint: str
    loader_id: str
    loader_version: str
    module_token: str
    load_generation: int
    symbols: Tuple[JitResolvedSymbol, ...]
    schema_version: int = JIT_MODULE_LOAD_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JIT_MODULE_LOAD_VERSION:
            raise SchemaError("unsupported JIT module load receipt version")
        if not str(self.spec_name):
            raise SchemaError("JIT loaded module spec_name must be non-empty")
        if not _IDENTIFIER.fullmatch(str(self.loader_id)):
            raise SchemaError("invalid JIT module loader_id")
        for name in ("loader_version", "module_token"):
            value = str(getattr(self, name))
            if not value or any(item in value for item in ("\x00", "\n", "\r")):
                raise SchemaError("JIT module %s must be safe and non-empty" % name)
            object.__setattr__(self, name, value)
        for name in (
            "spec_fingerprint",
            "cache_record_fingerprint",
            "artifact_verification_fingerprint",
            "artifact_fingerprint",
        ):
            _require_hash(name, getattr(self, name))
        if (
            not isinstance(self.load_generation, int)
            or isinstance(self.load_generation, bool)
            or self.load_generation < 1
        ):
            raise SchemaError("JIT module load generation must be positive")
        symbols = tuple(self.symbols)
        if not symbols or any(not isinstance(item, JitResolvedSymbol) for item in symbols):
            raise SchemaError("JIT loaded module must contain resolved symbols")
        names = tuple(item.name for item in symbols)
        if len(set(names)) != len(names):
            raise SchemaError("JIT loaded module symbols must be unique")
        object.__setattr__(self, "spec_name", str(self.spec_name))
        object.__setattr__(self, "loader_id", str(self.loader_id))
        object.__setattr__(self, "symbols", tuple(sorted(symbols, key=lambda item: item.name)))

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "spec_name": self.spec_name,
            "spec_fingerprint": self.spec_fingerprint,
            "cache_record_fingerprint": self.cache_record_fingerprint,
            "artifact_verification_fingerprint": self.artifact_verification_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "loader_id": self.loader_id,
            "loader_version": self.loader_version,
            "module_token": self.module_token,
            "load_generation": self.load_generation,
            "symbols": [item.to_dict() for item in self.symbols],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate(
        self,
        record: JitCacheRecord,
        verification: JitArtifactVerification,
        required_entry_points: Tuple[str, ...],
    ) -> None:
        if not isinstance(record, JitCacheRecord):
            raise TypeError("record must be JitCacheRecord")
        if not isinstance(verification, JitArtifactVerification):
            raise TypeError("verification must be JitArtifactVerification")
        verification.validate_record(record)
        required = tuple(str(item) for item in required_entry_points)
        if not required or any(not _SYMBOL.fullmatch(item) for item in required):
            raise SchemaError("required JIT entry points are invalid")
        if len(set(required)) != len(required):
            raise SchemaError("required JIT entry points must be unique")
        if (
            self.spec_name != record.spec_name
            or self.spec_fingerprint != record.spec_fingerprint
            or self.cache_record_fingerprint != record.fingerprint
            or self.artifact_verification_fingerprint != verification.fingerprint
            or self.artifact_fingerprint != record.artifact.fingerprint
            or tuple(item.name for item in self.symbols) != tuple(sorted(required))
        ):
            raise SchemaError("JIT loaded module does not bind artifact and symbols")


@dataclass(frozen=True)
class JitLoadedModule:
    """Validated receipt plus an opaque loader-owned module object."""

    receipt: JitModuleLoadReceipt
    opaque_module: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, JitModuleLoadReceipt):
            raise TypeError("receipt must be JitModuleLoadReceipt")
        if self.opaque_module is None:
            raise SchemaError("JIT loaded module opaque object must be present")

    @property
    def fingerprint(self) -> str:
        return self.receipt.fingerprint


@runtime_checkable
class JitModuleLoader(Protocol):
    """Private module loader supplied by an authorized runtime integration."""

    loader_id: str
    loader_version: str

    def load(
        self,
        record: JitCacheRecord,
        verification: JitArtifactVerification,
        required_entry_points: Tuple[str, ...],
    ) -> JitLoadedModule:
        """Load and resolve exact symbols; never called by the model user."""


def _relative_file(directory: Path, locator: str) -> Path:
    path = PurePosixPath(locator)
    if (path.is_absolute() or ".." in path.parts or str(path) != locator
            or "\\" in locator or locator == "."):
        raise SchemaError("invalid Triton metadata path: %r" % locator)
    resolved = (directory / locator).resolve(strict=True)
    if not resolved.is_relative_to(directory.resolve()):
        raise SchemaError("Triton metadata path escapes artifact root")
    return resolved


def _triton_files(directory: Path, artifact: ArtifactRef, spec_fingerprint: str):
    data = json.loads((directory / "triton.json").read_text())
    if (data["schema_version"] != 1 or data["spec_fingerprint"] != spec_fingerprint
            or data["entry_point"] != artifact.entry_point):
        raise SchemaError("Triton load metadata does not bind the compilation unit")
    group = {name: str(_relative_file(directory, path))
             for name, path in data["metadata_group"].items()}
    metadata_paths = [path for name, path in group.items() if name.endswith(".json")]
    binary_paths = [path for name, path in group.items() if name.endswith(".npubin")]
    if len(metadata_paths) != 1 or len(binary_paths) != 1:
        raise SchemaError("Triton metadata must identify exactly one declared npubin and metadata file")
    artifact.verify_bytes(Path(binary_paths[0]).read_bytes())
    metadata = json.loads(Path(metadata_paths[0]).read_text())
    if (metadata["name"] != artifact.entry_point or metadata["kernel_name"] != artifact.entry_point
            or metadata["hash"] != data["hash"] or metadata["target"] != data["target"]):
        raise SchemaError("Triton compiler metadata entry/hash/target mismatch")
    mode = data.get("binary_mode", "single")
    if mode not in {"single", "mix"} or ((mode == "mix") != (metadata["mix_mode"] == "mix")):
        raise SchemaError("Triton binary mode does not match compiler metadata")
    for name in ("npu_utils", "launcher"):
        path = _relative_file(directory, data[name])
        # These are CPU Python extensions, unlike the device npubin.
        if path.read_bytes()[:4] != b"\x7fELF":
            raise SchemaError("Triton %s is not an ELF extension" % name)
    return data, group


_triton_load_lock = RLock()


class _TritonExecutable:
    """Keep Triton's CompiledKernel/extensions alive and supply its stream ABI."""

    def __init__(self, kernel, extensions, data, device):
        self.kernel = kernel
        self.extensions = extensions
        self.data = data
        self.device = device

    def __getitem__(self, grid):
        grid = tuple(grid)
        if not 1 <= len(grid) <= 3 or any(not isinstance(n, int) or n <= 0 for n in grid):
            raise ValueError("Triton launch grid must contain 1 to 3 positive integers")
        grid = grid + (1,) * (3 - len(grid))

        def launch(*args, stream=None):
            import torch

            if torch.npu.current_device() != self.device:
                raise RuntimeError("Triton module must launch on its loaded NPU device")
            constants = self.data["constexprs"]
            runtime_names = [name for name in self.data["arg_names"] if name not in constants]
            if len(args) != len(runtime_names):
                raise TypeError("Triton entry expects runtime arguments %s" % runtime_names)
            values = dict(zip(runtime_names, args))
            values.update(constants)
            if stream is None:
                stream = torch.npu.current_stream(self.device).npu_stream
            return self.kernel[grid](*(values[name] for name in self.data["arg_names"]), stream=stream)

        return launch


class DefaultJitModuleLoader:
    """Load npubin through Triton's prebuilt runtime, or an explicit host .so.

    The Triton adapter targets the experimentally verified triton-ascend 3.6
    CompiledKernel ABI. It restores compiler metadata and prebuilt launchers;
    it never imports the original kernel source or invokes a compiler.
    """

    loader_id = "flashinfer.native"
    loader_version = "1"

    def __init__(self, store: DiskJitBuildStore, *, max_entries: int = 128) -> None:
        if not isinstance(store, DiskJitBuildStore):
            raise TypeError("default loader requires DiskJitBuildStore")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.store = store
        self.max_entries = max_entries
        self._loaded = OrderedDict()
        self._lock = RLock()
        self._generation = 0

    def _load_triton(self, directory, record):
        import importlib.util
        import torch
        import torch_npu
        import triton
        from triton.compiler.compiler import CompiledKernel
        from triton.backends.ascend.driver import wrap_handle_tensordesc

        data, group = _triton_files(directory, record.artifact, record.spec_fingerprint)
        versions = {"triton": triton.__version__, "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__, "python_abi": sys.implementation.cache_tag}
        if data["versions"] != versions or not triton.__version__.startswith("3.6."):
            raise RuntimeError("Triton runtime ABI/version mismatch: built=%r, current=%r" % (data["versions"], versions))
        device = torch.npu.current_device()
        if torch.npu.get_device_name(device) != data["target"]["arch"]:
            raise RuntimeError("Triton target differs from the current NPU device")
        from ._triton_abi import verify_triton_symbols
        verify_triton_symbols((directory / record.artifact.locator).read_bytes(),
                              record.artifact.entry_point, data.get("binary_mode", "single"))
        # CompiledKernel accepts a launch-source description; compilation and
        # the original Python source are deliberately absent on a disk hit.
        source = SimpleNamespace(signature=data["signature"], constants=data["constexprs"])
        kernel = CompiledKernel(source, group, data["hash"])

        def extension(name, path):
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module

        # This Triton launcher's module initializer locates npu_utils relative
        # to TRITON_CACHE_DIR. Once imported it retains function pointers; launch
        # does not consult this environment variable. Serialize this short scope.
        with _triton_load_lock:
            previous = os.environ.get("TRITON_CACHE_DIR")
            try:
                os.environ["TRITON_CACHE_DIR"] = str(directory / "triton-cache")
                utils = extension("npu_utils", str(_relative_file(directory, data["npu_utils"])))
                launcher = extension("__triton_launcher", str(_relative_file(directory, data["launcher"])))
            finally:
                if previous is None:
                    os.environ.pop("TRITON_CACHE_DIR", None)
                else:
                    os.environ["TRITON_CACHE_DIR"] = previous
        handles = utils.load_kernel_binary(kernel.metadata.kernel_name, kernel.kernel,
                                          kernel.metadata.shared, device, kernel.metadata.mix_mode)
        kernel.module, kernel.function, kernel.n_regs, kernel.n_spills, kernel.n_max_threads = handles
        if not kernel.module or not kernel.function:
            raise LookupError("Triton runtime did not resolve device symbol %r" % record.artifact.entry_point)
        signature = {index: data["signature"][name] for index, name in enumerate(data["arg_names"])}
        kernel._run = wrap_handle_tensordesc(launcher.launch, signature)
        return _TritonExecutable(kernel, (utils, launcher), data, device), device

    def load(self, record, verification, required_entry_points):
        verification.validate_record(record)
        required = tuple(required_entry_points)
        if not required or any(not _SYMBOL.fullmatch(name) for name in required) or len(set(required)) != len(required):
            raise SchemaError("required JIT entry points are invalid")
        directory = None
        kernel_id = record.spec_name
        try:
            directory = self.store.directory(record)
            kernel_id = json.loads((directory / "manifest.json").read_text())["spec"]["kernel_id"]
            record.artifact.verify_file(directory)
            device = None
            if record.artifact.format == ArtifactFormat.TRITON_NPUBIN:
                import torch
                import torch_npu

                device = torch.npu.current_device()
            key = (record.fingerprint, verification.fingerprint, required, device)
            with self._lock:
                if key in self._loaded:
                    self._loaded.move_to_end(key)
                    return self._loaded[key]
                if record.artifact.entry_point not in required:
                    raise LookupError("artifact entry %r does not match required symbols %r"
                                      % (record.artifact.entry_point, required))
                if record.artifact.format == ArtifactFormat.TRITON_NPUBIN:
                    if required != (record.artifact.entry_point,):
                        raise LookupError("a Triton unit exports exactly its declared device entry")
                    opaque, _ = self._load_triton(directory, record)
                elif record.artifact.format == ArtifactFormat.SHARED_LIBRARY:
                    import ctypes

                    opaque = ctypes.CDLL(str(directory / record.artifact.locator))
                    for name in required:
                        try:
                            getattr(opaque, name)
                        except AttributeError as error:
                            raise LookupError("shared library has no symbol %r" % name) from error
                elif record.artifact.format == ArtifactFormat.ASCENDC_OP_PACKAGE:
                    from .op_package import load_op_package

                    opaque = load_op_package(directory, target_soc=record.artifact.target_soc)
                    if required != (opaque.recipe.host_entry,):
                        raise LookupError("op-package ACLNN entry does not match required symbol")
                else:
                    raise RuntimeError("no default loader for artifact format %s" % record.artifact.format.value)
                self._generation += 1
                token = "%s:%s:%d:%s" % (record.fingerprint, os.getpid(), self._generation, device)
                receipt = JitModuleLoadReceipt(record.spec_name, record.spec_fingerprint,
                    record.fingerprint, verification.fingerprint, record.artifact.fingerprint,
                    self.loader_id, self.loader_version, token, self._generation,
                    tuple(JitResolvedSymbol(name, token + ":" + name) for name in required))
                receipt.validate(record, verification, required)
                loaded = JitLoadedModule(receipt, opaque)
                self._loaded[key] = loaded
                while len(self._loaded) > self.max_entries:
                    self._loaded.popitem(last=False)
                return loaded
        except Exception as error:
            logs = self.store.root / ".logs"
            logs.mkdir(parents=True, exist_ok=True)
            log_path = logs / (record.spec_fingerprint + "-load-" + uuid.uuid4().hex + ".log")
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            error_type = JitSymbolError if isinstance(error, LookupError) else JitLoadError
            raise error_type("%s; artifact=%s; required=%r" % (error, directory, required),
                kernel_id=kernel_id, spec_fingerprint=record.spec_fingerprint,
                log_path=log_path) from error


__all__ = [
    "JIT_MODULE_LOAD_VERSION",
    "JitLoadedModule",
    "JitModuleLoadReceipt",
    "JitModuleLoader",
    "JitResolvedSymbol",
    "DefaultJitModuleLoader",
]
