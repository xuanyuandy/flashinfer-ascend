"""Legacy input/hardware/toolchain helper (not a complete build identity).

key = sha256( 输入签名 × 硬件签名 × 工具链指纹 )。
host 环境无工具链时对应项为 ``"none"``——key 依然稳定，
满足 §14.1 单测要求（相同输入必同 key，任一组成变化必不同 key）。
T18B 的磁盘缓存使用 JitSpec.fingerprint，覆盖 kernel/source/options 等维度。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from ..runtime.hardware import HardwareSignature
from ..runtime.signature import InputSignature
from ..version import __version__

# cache key 结构版本：payload 结构变化时递增，旧 key 自动失效
_SCHEMA_VERSION = 1


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "none"


def _ascendc_toolkit_version() -> str:
    # CANN toolkit 版本由安装环境决定；host 开发机通常无安装
    return os.environ.get("ASCEND_TOOLKIT_VERSION", "none")


def toolchain_fingerprint() -> str:
    """工具链指纹：Ascend C 编译器版本 + Triton 版本 + codegen 版本。"""
    return "ascendc=%s;triton=%s;codegen=%s" % (
        _ascendc_toolkit_version(),
        _module_version("triton"),
        __version__,
    )


def build_cache_key(
    signature: InputSignature,
    hardware: HardwareSignature,
    toolchain: Optional[str] = None,
) -> str:
    """旧的输入侧指纹；真实编译缓存必须使用完整的 JitSpec.fingerprint。"""
    payload = {
        "input": signature.to_dict(),
        "hardware": {
            "soc_model": hardware.soc_model,
            "aicore_count": hardware.aicore_count,
        },
        "toolchain": toolchain
        if toolchain is not None
        else toolchain_fingerprint(),
        "schema": _SCHEMA_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
