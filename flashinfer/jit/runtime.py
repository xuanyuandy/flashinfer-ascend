"""Lazy AOT execution bindings for ordinary runtime descriptors.

An operator registration supplies a trusted frozen spec and its ABI adapter.
Public APIs keep using descriptor.runner after dispatch; the wrapper discovers
installed bytes only on first execution. Neither registration nor an AOT hit
probes sources, a toolchain, or a compiler. There are no operator-name switches.
"""

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from threading import RLock
from typing import Optional

from flashinfer import config
from flashinfer.runtime.schema import Backend, CompileMode, KernelDescriptor, SchemaError

from .core import JitSpec
from .env import JitCompilationPolicy


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeProvenance:
    """Evidence of the artifact actually bound, independent of dispatch's tier."""

    source: str
    directory: Path
    spec_fingerprint: str
    fallback_reason: Optional[str] = None


class AotKernelRunner:
    """Resolve once, then retain the module and its callable for process lifetime.

    Search indexes can be invalidated, but an already bound runner retains its
    loaded immutable generation. Replacing on-disk bytes does not unload CANN;
    deploying another package identity requires a fresh process. The existing
    loader enforces that restriction, and its error is never a JIT fallback.

    runner_factory(module, spec) adapts the operator's ABI to the public API.
    module is load_op_package's OpPackageModule for AOT, or build_and_load's
    JitLoadedModule for JIT. jit_backend_factory() is invoked only on a missing
    AOT artifact with ENABLED policy; it may resolve sources and toolchains.
    """

    def __init__(self, descriptor, *, spec, runner_factory, jit_backend_factory=None,
                 store=None, policy=None, index=None, package_peers=()):
        if not isinstance(descriptor, KernelDescriptor) or not isinstance(spec, JitSpec):
            raise TypeError("AOT binding requires KernelDescriptor and JitSpec")
        if descriptor.runner is not None:
            raise SchemaError("AOT binding must not overwrite an existing runner")
        if not (descriptor.backend == Backend.ASCENDC_AOT or (
                descriptor.backend == Backend.ASCENDC and descriptor.compile_mode == CompileMode.AOT)):
            raise SchemaError("AOT op-package binding requires an Ascend C AOT descriptor")
        spec.validate_descriptor(descriptor)
        if dict(spec.input_fingerprints).get("kernel") != descriptor.fingerprint:
            raise SchemaError("AOT spec must bind the descriptor fingerprint, including its ABI")
        from .op_package import OP_PACKAGE_ADAPTER_HASH

        if (spec.generator_id != "ascendc" or spec.output_path != "op-package.json"
                or dict(spec.input_fingerprints).get("output_adapter") != OP_PACKAGE_ADAPTER_HASH):
            raise SchemaError("AOT binding requires an Ascend C op-package spec")
        if not callable(runner_factory) or (jit_backend_factory is not None
                                           and not callable(jit_backend_factory)):
            raise TypeError("AOT runner/backend factories must be callable")
        self.descriptor = descriptor
        self.spec = spec
        self.runner_factory = runner_factory
        self.jit_backend_factory = jit_backend_factory
        self.store = store
        self.policy = None if policy is None else JitCompilationPolicy(policy)
        self.index = index
        self.package_peers = tuple(package_peers)
        for peer_descriptor, peer_spec in self.package_peers:
            peer_spec.validate_descriptor(peer_descriptor)
            if dict(peer_spec.input_fingerprints).get("kernel") != peer_descriptor.fingerprint:
                raise SchemaError("AOT companion spec must bind its trusted descriptor")
        self._lock = RLock()
        self._runner = None
        self._module = None
        self._provenance = None

    @property
    def provenance(self):
        """None until a verified module has been adapted successfully."""
        return self._provenance

    def resolve(self):
        """Only AotMissingError authorizes the explicitly enabled JIT branch."""
        from .discovery import AotLoadError, AotMissingError, get_op_package_index

        with self._lock:
            if self._runner is not None:
                return self._runner
            index = self.index if self.index is not None else get_op_package_index()
            try:
                match = index.find(self.spec, descriptor=self.descriptor,
                                   target_soc=self.spec.target_soc)
            except AotMissingError as missing:
                policy = self.policy
                if policy is None:
                    policy = (JitCompilationPolicy.DISABLED if config.jit_disabled()
                              else JitCompilationPolicy.ENABLED)
                # CACHE_ONLY also rejects missing distribution artifacts. This
                # runtime contract does not substitute an unrelated JIT cache
                # for an absent AOT installation under a no-fallback policy.
                if policy != JitCompilationPolicy.ENABLED or self.jit_backend_factory is None:
                    detail = ("no JIT backend factory registered" if self.jit_backend_factory is None
                              else "JIT fallback is not permitted")
                    raise AotMissingError(
                        missing.spec, missing.target_soc, missing.roots, missing.rejections,
                        detail=f"compilation policy={policy.value}; {detail}",
                    ) from missing
                from .build import build_and_load
                from .cache import DiskJitBuildStore, JitCacheIndex

                store = self.store if self.store is not None else DiskJitBuildStore()
                cache = JitCacheIndex()
                _logger.info("AOT missing for %s; JIT fallback: %s", self.spec.kernel_id, missing)
                module = build_and_load(self.spec, backend=self.jit_backend_factory(),
                                        store=store, cache=cache, policy=policy)
                record = cache.lookup(self.spec)
                provenance = RuntimeProvenance("jit", store.directory(record),
                                               self.spec.fingerprint, str(missing))
                runner = self.runner_factory(module, self.spec)
            else:
                # Integrity errors, dlopen/symbol errors and process-scoped
                # package identity conflicts propagate; none enter the except
                # AotMissingError branch above. CANN runtime version comparison
                # remains a known gap in load_op_package, not claimed here.
                companions = []
                for peer_descriptor, peer_spec in self.package_peers:
                    if peer_spec.kernel_id == self.spec.kernel_id or peer_spec.target_soc != self.spec.target_soc:
                        continue
                    try:
                        companion = index.find(peer_spec, descriptor=peer_descriptor,
                                               target_soc=peer_spec.target_soc)
                    except AotMissingError:
                        # A CONFIG/single-op wheel may omit an optional peer.
                        # Preparing a group must never compile another operator.
                        continue
                    companions.append(companion)
                module = match.load(companions=tuple(companions)) if companions else match.load()
                provenance = RuntimeProvenance("aot", match.directory, self.spec.fingerprint)
                try:
                    runner = self.runner_factory(module, self.spec)
                    if not callable(runner):
                        raise TypeError("AOT runner_factory must return a callable")
                except Exception as error:
                    raise AotLoadError(match.directory, error) from error
            if not callable(runner):
                raise TypeError("AOT runner_factory must return a callable")
            self._module = module
            self._runner = runner
            self._provenance = provenance
            _logger.info("Bound %s kernel=%s spec=%s directory=%s", provenance.source,
                         self.spec.kernel_id, provenance.spec_fingerprint, provenance.directory)
            return runner

    def __call__(self, *args, **kwargs):
        # Runtime/CANN launch errors retain their original type and traceback.
        # In particular they must never be interpreted as an artifact miss.
        return self.resolve()(*args, **kwargs)


def bind_aot_kernel(descriptor, *, spec, runner_factory, jit_backend_factory=None,
                    store=None, policy=None, index=None, package_peers=()):
    """Return a descriptor ready for ordinary KernelRegistry registration.

    This is an operator-author registration hook, not an application setup
    requirement: installed operators should register it alongside their static
    descriptor. Their users call the existing public API with no package paths.
    Optional peers supply trusted declarations for a shared CANN activation;
    missing peers are not compiled. This performs no I/O and preserves the frozen
    descriptor fingerprint because runners are not serialized ABI fields.
    """
    return replace(descriptor, runner=AotKernelRunner(
        descriptor, spec=spec, runner_factory=runner_factory,
        jit_backend_factory=jit_backend_factory, store=store, policy=policy, index=index,
        package_peers=package_peers))


__all__ = ["AotKernelRunner", "RuntimeProvenance", "bind_aot_kernel"]
