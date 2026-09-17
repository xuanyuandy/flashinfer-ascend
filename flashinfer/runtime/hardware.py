"""硬件签名与探测（设计文档 §4）。

签名只保留 **SoC 系列型号 + AI Core 数量**两个字段（§4.1）：
同系列不同 SKU（如 910B1 / 910B4）靠 aicore_count 区分；
软件栈版本（CANN / 驱动）由环境校验承担，不进签名（§4.4）；
``ASCEND_RT_VISIBLE_DEVICES`` 只影响可见卡数，不进签名（§4.2）。

无 NPU 环境（host 开发机）时降级为 ``unknown / 0``——
dispatch 决策、cache key 依然可运行（§14.1 host 单测的前提）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareSignature:
    """当前硬件的静态特征快照，不可变、可哈希。"""

    soc_model: str = "unknown"  # 系列型号，如 "Ascend910B"；unknown 表示无 NPU
    aicore_count: int = 0       # 0 表示未知

    @property
    def is_npu_available(self) -> bool:
        return self.soc_model != "unknown"


_cache_lock = threading.Lock()
_cached: Optional[HardwareSignature] = None

# 探测只精确到系列型号（§4.1），更细的 SKU 差异由 aicore_count 承担
_SOC_PREFIXES = ("Ascend910B", "Ascend910C", "Ascend310P", "Ascend910", "Ascend310")


def _normalize_soc_model(device_name: str) -> str:
    """``"Ascend910B3"`` → ``"Ascend910B"``：只保留系列型号。"""
    name = device_name.replace(" ", "")
    for prefix in _SOC_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return name or "unknown"


def _probe() -> HardwareSignature:
    """通过 torch_npu 探测硬件；任何一步失败都降级为 unknown，不抛错。"""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception:
        return HardwareSignature()
    try:
        if not torch.npu.is_available():
            return HardwareSignature()
        soc = _normalize_soc_model(str(torch.npu.get_device_name(0)))
        aicore = 0
        try:
            props = torch.npu.get_device_properties(0)
            aicore = int(getattr(props, "multi_processor_count", 0))
        except Exception:
            pass
        return HardwareSignature(soc_model=soc, aicore_count=aicore)
    except Exception:
        return HardwareSignature()


def current_hardware() -> HardwareSignature:
    """进程内一次构建并缓存（§4.3：探测仅在首次调用时发生）。"""
    global _cached
    if _cached is None:
        with _cache_lock:
            if _cached is None:
                _cached = _probe()
    return _cached


def _reset_hardware_cache_for_testing(
    signature: Optional[HardwareSignature] = None,
) -> None:
    """测试注入用：重置为 None（下次重新探测）或替换为指定签名。"""
    global _cached
    with _cache_lock:
        _cached = signature
