"""runtime 层对外导出：签名 / 硬件 / 契约 / 注册表 / dispatcher / Plan。"""

from .signature import InputSignature, build_signature
from .hardware import HardwareSignature, current_hardware
from .schema import (
    Backend,
    CapabilitySpec,
    CompileMode,
    DispatchError,
    KernelDescriptor,
)
from .registry import (
    CUSTOM_KERNEL_PREFIX,
    KernelRegistry,
    get_global_registry,
    register_custom_kernel,
)
from .dispatch import (
    CandidateReport,
    DispatchDecision,
    clear_dispatch_cache,
    dispatch,
    explain_dispatch,
)
from .plan import Plan

__all__ = [
    "Backend",
    "CUSTOM_KERNEL_PREFIX",
    "CandidateReport",
    "CapabilitySpec",
    "CompileMode",
    "DispatchDecision",
    "DispatchError",
    "HardwareSignature",
    "InputSignature",
    "KernelDescriptor",
    "KernelRegistry",
    "Plan",
    "build_signature",
    "clear_dispatch_cache",
    "current_hardware",
    "dispatch",
    "explain_dispatch",
    "get_global_registry",
    "register_custom_kernel",
]
