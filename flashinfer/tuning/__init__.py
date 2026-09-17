"""tuning 子系统（设计文档 §9，进阶特性，后期启用）。

版本定位：tuning 是进阶优化特性。无 tuning 记录时 dispatch 走确定性
启发式（§5.4.3），功能完整可用——M0~M4 均不依赖本子系统。

当前仅提供查询接口占位（恒返回 None，即"未命中"），
保证 dispatch 的"tuning 优先"分支有明确落点。
"""

from __future__ import annotations

from typing import Optional

from ..runtime.hardware import HardwareSignature
from ..runtime.signature import InputSignature


def query_tuning_record(
    signature: InputSignature, hardware: HardwareSignature
) -> Optional[str]:
    """按 (op, workload bucket, capability fingerprint) 查询冠军 kernel_id。

    进阶特性（§9）：发布期实测数据库与本地 tuning 库均未接入，
    当前恒返回 None（未命中），dispatch 落确定性启发式。
    """
    return None
