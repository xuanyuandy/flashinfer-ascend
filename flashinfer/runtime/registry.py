"""kernel 注册表：注册、冻结与自定义 kernel 隔离（设计文档 §10.1）。

- 内置 kernel 在初始化期注册，随后 ``freeze()``——冻结后再注册内置
  kernel 抛错（§10.1：import 后新 kernel 可见期与初始注册期错开）。
- 经本仓验证的外部仓 kernel 走 ``register_external()``，允许冻结后接入；
  强制 ``backend=EXTERNAL``，使用外部仓自己的命名空间，默认参与 auto。
- 自定义 kernel 走 ``register_custom()``，不受冻结限制，但强制
  ``backend=CUSTOM`` 且 ``kernel_id`` 以 ``custom/`` 为命名空间前缀（§10.1）；
  与任何已有 kernel 同名抛错，不允许覆盖（§10.5.1、§12）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .schema import Backend, KernelDescriptor

CUSTOM_KERNEL_PREFIX = "custom/"

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_REGISTRY: Optional["KernelRegistry"] = None


@dataclass(frozen=True)
class CompatibilityReport:
    kernel_id: str
    reasons: Tuple[str, ...]


class KernelRegistry:
    """运行期可见 kernel 的唯一事实源。"""

    def __init__(self, descriptors: Iterable[KernelDescriptor] = ()) -> None:
        self._descriptors: Dict[str, KernelDescriptor] = {}
        self._frozen = False
        # 注册计数：dispatch 决策缓存 key 的组成部分，注册变化自动使旧缓存失效
        self._revision = 0
        self._lock = threading.Lock()
        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def register(self, descriptor: KernelDescriptor) -> None:
        """注册内置 kernel（仅初始化期），外部仓与 custom 使用专用入口。"""
        with self._lock:
            if self._frozen:
                raise RuntimeError("registry 已冻结，内置 kernel 不可再注册")
            if descriptor.backend == Backend.CUSTOM:
                raise ValueError(
                    "内置 kernel 不得使用 custom backend，请走 register_custom()"
                )
            if descriptor.backend == Backend.EXTERNAL:
                raise ValueError(
                    "外部仓 kernel 请走 register_external()"
                )
            self._insert(descriptor)

    def register_external(self, descriptor: KernelDescriptor) -> None:
        """接入经本仓验证的外部仓实现，允许冻结后调用并使调度缓存失效。

        kernel_id 使用仓库自有前缀（如 ``vllm-ascend/``），不得占用
        backend 名称对应的框架保留前缀，尤其是 ``custom/``。
        """
        with self._lock:
            if descriptor.backend != Backend.EXTERNAL:
                raise ValueError("外部仓 kernel 的 backend 必须为 external")
            namespace, separator, name = descriptor.kernel_id.partition("/")
            if not namespace or not separator or not name:
                raise ValueError(
                    "外部仓 kernel_id 必须使用仓库自有命名空间前缀，"
                    "例如 'vllm-ascend/op/kernel'"
                )
            if namespace in {backend.value for backend in Backend}:
                raise ValueError(
                    "外部仓 kernel_id 不得使用框架保留前缀 %r"
                    % (namespace + "/")
                )
            self._insert(descriptor)

    def register_custom(self, descriptor: KernelDescriptor) -> None:
        """注册自定义 kernel（§10.1），允许在冻结后调用。"""
        with self._lock:
            if descriptor.backend != Backend.CUSTOM:
                raise ValueError("自定义 kernel 的 backend 必须为 custom（§5.2）")
            if not descriptor.kernel_id.startswith(CUSTOM_KERNEL_PREFIX):
                raise ValueError(
                    "自定义 kernel_id 必须以 %r 为前缀（命名空间隔离，§10.1）"
                    % CUSTOM_KERNEL_PREFIX
                )
            self._insert(descriptor)

    def _insert(self, descriptor: KernelDescriptor) -> None:
        if (
            descriptor.backend != Backend.CUSTOM
            and descriptor.kernel_id.startswith(CUSTOM_KERNEL_PREFIX)
        ):
            raise ValueError("custom/ 命名空间仅供 custom backend 使用")
        if descriptor.kernel_id in self._descriptors:
            raise ValueError(
                "duplicate kernel_id 重复：%s（不允许覆盖，§12）" % descriptor.kernel_id
            )
        self._descriptors[descriptor.kernel_id] = descriptor
        self._revision += 1

    def get(self, kernel_id: str) -> KernelDescriptor:
        try:
            return self._descriptors[kernel_id]
        except KeyError:
            raise KeyError("未注册的 kernel_id：%s" % kernel_id)

    def descriptors(self, op: Optional[str] = None) -> Tuple[KernelDescriptor, ...]:
        values: Iterable[KernelDescriptor] = self._descriptors.values()
        if op is not None:
            values = tuple(d for d in values if d.op == op)
        return tuple(sorted(values, key=lambda d: d.kernel_id))

    def explain(self, workload, capability):
        """Report manifest constraints without importing or executing a provider."""
        return tuple(
            CompatibilityReport(
                d.kernel_id,
                d.constraints.unsupported_reasons(workload, capability)
                + d.unsupported_target_reasons(capability.soc_version),
            )
            for d in self.descriptors(workload.op)
        )

    def select(self, workload, capability, *, backend=None,
               allow_reference=False, tuned_kernel_ids=()):
        """Select a manifest descriptor for the existing build_plan API.

        Tensor frontends continue to use runtime.dispatch; this path consumes
        WorkloadSpec/DeviceCapability, as required by manifest clients.
        """
        from .dispatch import DispatchError

        policy = None if backend in (None, "auto") else Backend(backend)
        candidates = []
        rejected = []
        for report in self.explain(workload, capability):
            descriptor = self.get(report.kernel_id)
            reasons = list(report.reasons)
            if policy is not None and descriptor.backend != policy:
                reasons.append("forced backend does not match")
            if descriptor.backend == Backend.REFERENCE and not allow_reference:
                reasons.append("reference requires explicit opt-in")
            if reasons:
                rejected.append(f"{descriptor.kernel_id}: {', '.join(reasons)}")
            else:
                candidates.append(descriptor)
        if not candidates:
            raise DispatchError("no compatible kernel; " + "; ".join(rejected))
        tuning = {kernel_id: i for i, kernel_id in enumerate(tuned_kernel_ids)}
        return min(candidates, key=lambda d: (
            tuning.get(d.kernel_id, len(tuning)),
            d.backend == Backend.REFERENCE,
            -d.priority,
            d.kernel_id,
        ))


def _builtin_descriptors() -> Tuple[KernelDescriptor, ...]:
    """内置 kernel 注册表：reference + CANN aclnn + Triton（M1 起）。"""
    from ..cann import cann_descriptors
    from ..triton import triton_descriptors
    from . import reference_kernels
    from .._sampling import descriptors as sampling_descriptors
    from .._activation import descriptors as activation_descriptors
    from .._page import descriptors as page_descriptors
    from .._cascade import descriptors as cascade_descriptors

    return (
        reference_kernels.builtin_descriptors()
        + cann_descriptors()
        + triton_descriptors()
        + sampling_descriptors(Backend.REFERENCE)
        + activation_descriptors(Backend.REFERENCE)
        + page_descriptors(Backend.REFERENCE)
        + cascade_descriptors(Backend.REFERENCE)
    )


def get_global_registry() -> KernelRegistry:
    """进程级单例：首次访问时注册内置 kernel 并冻结。"""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_REGISTRY is None:
                registry = KernelRegistry(_builtin_descriptors())
                registry.freeze()
                _GLOBAL_REGISTRY = registry
    return _GLOBAL_REGISTRY


def register_custom_kernel(descriptor: KernelDescriptor) -> None:
    """用户自定义 kernel 的注册入口（§10.1）。"""
    get_global_registry().register_custom(descriptor)


def register_external_kernel(descriptor: KernelDescriptor) -> None:
    """经本仓验证的外部仓 kernel 接入入口（非用户自定义注册）。"""
    get_global_registry().register_external(descriptor)


def _reset_global_registry_for_testing() -> None:
    global _GLOBAL_REGISTRY
    with _GLOBAL_LOCK:
        _GLOBAL_REGISTRY = None
