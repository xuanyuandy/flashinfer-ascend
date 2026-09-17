"""JIT 子系统（设计文档 §8）：cache key、缓存与编译封装。

M0 提供 cache key 构建（§8.1）；进程内 LRU / 磁盘缓存目录布局 /
编译拉起（§8.3、§8.4）随首批自研 kernel（M1 起）接入。
"""

from .cache_key import build_cache_key, toolchain_fingerprint

__all__ = ["build_cache_key", "toolchain_fingerprint"]
