"""Plan：dispatch 结果的物化（设计文档 §11）。

Plan 不可变：选定 kernel、输入签名、硬件快照、workspace 规划。
复杂算子的 wrapper（如 BatchDecodeWithPagedKVCacheWrapper）持有 Plan
生命周期，run() 直接复用，不重复 dispatch（§3.3）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .dispatch import DispatchDecision
from .hardware import HardwareSignature
from .signature import InputSignature


@dataclass(frozen=True)
class Plan:
    signature: InputSignature
    hardware: HardwareSignature
    decision: DispatchDecision
    workspace_bytes: int = 0  # workspace 上界，M0 占位（§3.3 内存规划后续接入）

    @property
    def op(self) -> str:
        return self.signature.op

    @property
    def kernel_id(self) -> str:
        return self.decision.kernel_id


@dataclass(frozen=True)
class _WorkloadPlan:
    workload: object
    capability: object
    kernel: object
    workspace_bytes: int
    int_workspace_bytes: int

    @property
    def kernel_id(self):
        return self.kernel.kernel_id

    @property
    def fingerprint(self):
        from .schema import _fingerprint
        return _fingerprint(dict(
            workload=self.workload.fingerprint, capability=self.capability.fingerprint,
            kernel=self.kernel.fingerprint, workspace_bytes=self.workspace_bytes,
            int_workspace_bytes=self.int_workspace_bytes,
        ))

    def validate_runtime(self, workload, capability):
        from .schema import SchemaError
        if workload != self.workload or capability != self.capability:
            raise SchemaError("runtime workload/capability differs from plan")


def build_plan(registry, workload, capability, **selection):
    """Restore the explicit workload plan consumed by historical host clients."""
    kernel = registry.select(workload, capability, **selection)
    return _WorkloadPlan(workload, capability, kernel,
                         kernel.workspace.size_for(workload),
                         kernel.int_workspace.size_for(workload))
