"""配置项（设计文档 §13）：全部经环境变量读取。

不做进程内缓存——每次调用直读 ``os.environ``，测试可通过
``monkeypatch.setenv`` 即时改变行为。
"""

from __future__ import annotations

import os

_DEFAULT_CACHE_ROOT = os.path.join("~", ".cache", "flashinfer")
_DEFAULT_DISPATCH_ORDER = ("external", "cann", "aot", "jit", "reference")


def cache_dir() -> str:
    """JIT 缓存根目录。"""
    return os.path.expanduser(
        os.environ.get("FLASHINFER_NPU_CACHE_DIR", _DEFAULT_CACHE_ROOT)
    )


def default_backend() -> str:
    """全局默认 backend policy（``backend=None`` 时生效）。"""
    return os.environ.get("FLASHINFER_NPU_BACKEND", "auto")


def jit_disabled() -> bool:
    """禁用 JIT（仅 AOT / CANN 可用）。"""
    return os.environ.get("FLASHINFER_NPU_JIT_DISABLE", "0") == "1"


def jit_cache_size() -> int:
    """进程内 LRU 条目上限。"""
    return int(os.environ.get("FLASHINFER_NPU_JIT_CACHE_SIZE", "1024"))


def dispatch_cache_disabled() -> bool:
    """禁用 dispatch 决策缓存（§5.3.2）。"""
    return os.environ.get("FLASHINFER_NPU_DISABLE_DISPATCH_CACHE", "0") == "1"


def dispatch_order() -> tuple[str, ...]:
    """无 tuning 冠军时的档位顺序，按优先到后备排列。

    ``FLASHINFER_NPU_DISPATCH_ORDER`` 必须是 external,cann,aot,jit,reference
    的完整逗号分隔排列，每档恰好一次；允许分隔符两侧空白。
    cann 包含 ACLNN；aot/jit 包含自研及已 opt-in 的 custom，未声明
    AOT 的自研归 jit。非法值抛 ValueError，不缓存环境变量。
    """
    name = "FLASHINFER_NPU_DISPATCH_ORDER"
    raw = os.environ.get(name)
    if raw is None:
        return _DEFAULT_DISPATCH_ORDER
    order = tuple(item.strip() for item in raw.split(","))
    unknown = sorted(set(order) - set(_DEFAULT_DISPATCH_ORDER))
    missing = sorted(set(_DEFAULT_DISPATCH_ORDER) - set(order))
    duplicates = sorted(item for item in set(order) if order.count(item) > 1)
    if unknown or missing or duplicates:
        details = []
        if unknown:
            details.append(f"未知档位 {unknown!r}")
        if missing:
            details.append(f"缺少档位 {missing!r}")
        if duplicates:
            details.append(f"重复档位 {duplicates!r}")
        raise ValueError(
            f"{name}={raw!r} 无效：{'；'.join(details)}。"
            "须将 external,cann,aot,jit,reference 按优先顺序完整排列，"
            "逗号分隔且每档恰好一次。"
        )
    return order


def allow_custom_kernels() -> bool:
    """允许自定义 kernel 参与 auto 竞争（§10.5.2）。

    只影响 auto 候选池；显式 ``kernel_id=`` 指定永远可用。
    """
    return os.environ.get("FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS", "0") == "1"


def tuning_dir() -> str:
    """本地 tuning database 目录（§9，进阶特性）。"""
    return os.environ.get(
        "FLASHINFER_NPU_TUNING_DIR", os.path.join(cache_dir(), "tuning")
    )


def online_tuning_enabled() -> bool:
    """在线探索 tuning 开关（§9.4，进阶特性，默认关闭）。"""
    return os.environ.get("FLASHINFER_NPU_ONLINE_TUNING", "0") == "1"


def format_config() -> str:
    """Host-only diagnostics for the environment-backed runtime configuration."""
    return "\n".join(f"{name}: {value}" for name, value in (
        ("cache_dir", cache_dir()), ("backend", default_backend()),
        ("jit_disabled", jit_disabled()), ("jit_cache_size", jit_cache_size()),
        ("dispatch_order", ",".join(dispatch_order())),
        ("tuning_dir", tuning_dir()), ("online_tuning", online_tuning_enabled()),
    ))
