"""runtime 层对外导出：签名 / 硬件 / 契约 / 注册表 / dispatcher / Plan。"""

from .signature import InputSignature, build_signature
from .hardware import HardwareSignature, current_hardware
from .schema import (
    Backend,
    CapabilitySpec,
    CompileMode,
    DispatchError,
    GraphCapability,
    KernelDescriptor,
    ShapeRange,
)
from .registry import (
    CUSTOM_KERNEL_PREFIX,
    ImplementationStack,
    KernelRegistry,
    get_global_registry,
    kernel_registry_scope,
    register_custom_kernel,
    register_external_kernel,
)
from .dispatch import (
    CandidateReport,
    DispatchDecision,
    StackReport,
    clear_dispatch_cache,
    dispatch,
    explain_dispatch,
)
from .plan import Plan
from .schema import (
    ArtifactFormat, ArtifactKind, ArtifactRef, ArtifactVerificationError,
    CFieldABI, CPrimitive, CStructABI, DeviceCapability, KernelArgumentABI,
    KernelArgumentDirection, KernelArgumentPassing, KernelBinaryABI,
    KernelCapabilityBinding, KernelConstraints, KernelErrorABI, KernelErrorCodeABI,
    KernelLaunchABI, QuantSpec, SchemaError, SourceRef, WorkloadSpec, WorkspaceFormula,
)
from .manifest import load_kernel_manifest
from .plan import build_plan

__all__ = [
    "Backend",
    "CUSTOM_KERNEL_PREFIX",
    "CandidateReport",
    "CapabilitySpec",
    "CompileMode",
    "DispatchDecision",
    "DispatchError",
    "GraphCapability",
    "HardwareSignature",
    "InputSignature",
    "ImplementationStack",
    "KernelDescriptor",
    "KernelRegistry",
    "Plan",
    "ShapeRange",
    "StackReport",
    "build_signature",
    "clear_dispatch_cache",
    "current_hardware",
    "dispatch",
    "explain_dispatch",
    "get_global_registry",
    "kernel_registry_scope",
    "register_custom_kernel",
    "register_external_kernel",
    "ArtifactFormat", "ArtifactKind", "ArtifactRef", "ArtifactVerificationError",
    "CFieldABI", "CPrimitive", "CStructABI", "DeviceCapability", "KernelArgumentABI",
    "KernelArgumentDirection", "KernelArgumentPassing", "KernelBinaryABI",
    "KernelCapabilityBinding", "KernelConstraints", "KernelErrorABI", "KernelErrorCodeABI",
    "KernelLaunchABI", "QuantSpec", "SchemaError", "SourceRef", "WorkloadSpec", "WorkspaceFormula",
    "load_kernel_manifest", "build_plan",
]
