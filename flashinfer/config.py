"""配置项（设计文档 §13）：全部经环境变量读取。

不做进程内缓存——每次调用直读 ``os.environ``，测试可通过
``monkeypatch.setenv`` 即时改变行为。
"""

from __future__ import annotations

import os

_DEFAULT_CACHE_ROOT = os.path.join("~", ".cache", "flashinfer")


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
