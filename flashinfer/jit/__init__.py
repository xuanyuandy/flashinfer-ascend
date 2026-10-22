"""JIT 子系统（设计文档 §8）：声明、状态与后端协议。

T18A 提供统一编译单元、后端声明与只读产物映射；真实编译、装载、
磁盘缓存、文件锁和原子发布等待 T18B。导入不依赖编译器或其 SDK。
"""

from .cache_key import build_cache_key, toolchain_fingerprint
from .env import JitEnvironment, JitCompilationPolicy
from .core import JitSpec, JitSpecRegistry, JitSpecStatus, gen_jit_spec, jit_spec_registry
from .cache import (
    JitBuildStore, JitCacheIndex, JitCacheRecord, JitResolution, JitResolutionState,
    MissingJITCacheError, require_jit_cache_hit, resolve_jit_spec,
)
from .artifacts import (
    ConfiguredJitArtifactVerifier, JitArtifactPayloadReader, JitArtifactVerification,
    JitArtifactVerifier, verify_jit_cache_record_payload,
)
from .loading import (
    JitLoadedModule, JitModuleLoader, JitModuleLoadReceipt, JitResolvedSymbol,
)
from .backends import AscendCBackend, JitBackend, TritonBackend, declare_kernel
from .status import JitArtifactState, JitArtifactStatus, describe_artifact

__all__ = ["build_cache_key", "toolchain_fingerprint"]
__all__ += [name for name in globals() if name.startswith("Jit") or name.startswith("ConfiguredJit")]
__all__ += ["gen_jit_spec", "MissingJITCacheError", "require_jit_cache_hit",
            "resolve_jit_spec", "verify_jit_cache_record_payload", "jit_spec_registry",
            "AscendCBackend", "TritonBackend", "declare_kernel", "describe_artifact"]
