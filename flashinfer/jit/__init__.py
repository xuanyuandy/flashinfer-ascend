"""JIT declarations and explicit lazy building/loading. Import needs no SDK."""

from .cache_key import build_cache_key, toolchain_fingerprint
from .env import JitEnvironment, JitCompilationPolicy
from .core import JitSpec, JitSpecRegistry, JitSpecStatus, gen_jit_spec, jit_spec_registry
from .cache import (
    JitBuildStore, JitCacheIndex, JitCacheRecord, JitResolution, JitResolutionState,
    MissingJITCacheError, require_jit_cache_hit, resolve_jit_spec,
    DiskJitBuildStore, jit_cache_index,
)
from .artifacts import (
    ConfiguredJitArtifactVerifier, JitArtifactPayloadReader, JitArtifactVerification,
    JitArtifactVerifier, verify_jit_cache_record_payload,
)
from .loading import (
    JitLoadedModule, JitModuleLoader, JitModuleLoadReceipt, JitResolvedSymbol,
    DefaultJitModuleLoader,
)
from .backends import AscendCBackend, JitBackend, TritonBackend, declare_kernel
from .status import JitArtifactState, JitArtifactStatus, describe_artifact
from .build import build_and_load
from .errors import JitBuildError, JitCompilationError, JitLoadError, JitSymbolError
from .recipe import (
    AscendCBuildConfig, BuildDependency, BuildRecipe, ResolvedBuildRecipe,
    load_build_recipes, resolve_build_recipe,
)
from .ascendc import AscendCToolchain

__all__ = ["build_cache_key", "toolchain_fingerprint"]
__all__ += [name for name in globals() if name.startswith("Jit") or name.startswith("ConfiguredJit")]
__all__ += ["gen_jit_spec", "MissingJITCacheError", "require_jit_cache_hit",
            "resolve_jit_spec", "verify_jit_cache_record_payload", "jit_spec_registry",
            "AscendCBackend", "TritonBackend", "declare_kernel", "describe_artifact"]
__all__ += ["DiskJitBuildStore", "DefaultJitModuleLoader", "jit_cache_index", "build_and_load"]
__all__ += ["AscendCBuildConfig", "AscendCToolchain", "BuildDependency", "BuildRecipe",
            "ResolvedBuildRecipe", "load_build_recipes", "resolve_build_recipe"]
