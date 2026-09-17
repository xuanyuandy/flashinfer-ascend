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
