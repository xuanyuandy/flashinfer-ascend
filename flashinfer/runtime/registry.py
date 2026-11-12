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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

from .. import config
from .hardware import HardwareSignature
from .schema import Backend, CompileMode, GraphCapability, KernelDescriptor
from .signature import InputSignature

CUSTOM_KERNEL_PREFIX = "custom/"

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_REGISTRY: Optional["KernelRegistry"] = None
_SCOPED_REGISTRY = ContextVar("flashinfer_kernel_registry", default=None)


@dataclass(frozen=True)
class CompatibilityReport:
    kernel_id: str
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ImplementationStack:
    """同一来源的只读登记视图；不携带 capability 或额外的排序策略。

    ``stack_id`` 跨 op 稳定：内置来源为 cann / ascendc / triton /
    reference / custom；外部仓为 ``external:<仓库命名空间>``。历史
    ACLNN 与 CANN 合栈，Ascend C 的 AOT/JIT 合栈，编译方式不改变来源。
    ``descriptors`` 可由 ``KernelRegistry.stacks(op=...)`` 限定到某个 op。
    """

    stack_id: str
    descriptors: Tuple[KernelDescriptor, ...]


# Backend 是来源分类；栈身份另外包含 T6.2 已验证的外部仓命名空间。
_BUILTIN_STACK_IDS = {
    Backend.CANN: "cann", Backend.ACLNN: "cann",
    Backend.ASCENDC: "ascendc", Backend.ASCENDC_AOT: "ascendc",
    Backend.ASCENDC_JIT: "ascendc", Backend.TRITON: "triton",
    Backend.REFERENCE: "reference", Backend.CUSTOM: "custom",
}


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

    def graph_requirements(self, kernel_id: str) -> Tuple[GraphCapability, Tuple[str, ...]]:
        """返回 ``(入图性质, 待更新入参名元组)``，未知 ID 沿用 get 的 KeyError。

        仅查询静态声明，不代表经过入图验证，不绑定任务句柄或执行参数更新。
        参数名属于该 kernel；T11 负责将其绑定到录制任务的参数布局。
        """
        descriptor = self.get(kernel_id)
        return descriptor.graph_capability, descriptor.graph_param_updates

    def descriptors(self, op: Optional[str] = None) -> Tuple[KernelDescriptor, ...]:
        values: Iterable[KernelDescriptor] = self._descriptors.values()
        if op is not None:
            values = tuple(d for d in values if d.op == op)
        return tuple(sorted(values, key=lambda d: d.kernel_id))

    def stack_id_for(self, kernel_id: str) -> str:
        """从已登记的来源身份确定栈；不导入 provider 或探测编译器。"""
        descriptor = self.get(kernel_id)
        if descriptor.backend == Backend.EXTERNAL:
            # register_external 已检查该前缀，不能合并所有 EXTERNAL。
            return "external:" + descriptor.kernel_id.partition("/")[0]
        return _BUILTIN_STACK_IDS[descriptor.backend]

    def stacks(self, op: Optional[str] = None) -> Tuple[ImplementationStack, ...]:
        """按来源分组，返回稳定排序的快照，不复制或改写 descriptor。

        从登记记录派生，冻结后新接入的 external/custom 立即出现在视图中，
        仍由同一个 revision 使 dispatch 缓存失效。
        """
        groups: Dict[str, list] = {}
        for descriptor in self.descriptors(op):
            groups.setdefault(self.stack_id_for(descriptor.kernel_id), []).append(descriptor)
        return tuple(ImplementationStack(stack_id, tuple(descriptors))
                     for stack_id, descriptors in sorted(groups.items()))

    def explain(self, workload, capability):
        """Report input compatibility without importing or executing a provider.

        This remains policy-free: backend/custom/reference/JIT opt-ins and
        ordering belong to select(), unlike explain_dispatch's decision trace.
        """
        # Typed workload fields are authoritative if attributes repeat a name.
        # Keep other duplicate attributes: the predicate must check every value,
        # just as it does for InputSignature.specializations.
        specializations = tuple(
            (key, value) for key, value in workload.attributes
            if key not in ("causal", "pos_encoding", "deterministic")
        ) + (
            ("causal", str(workload.causal)),
            ("pos_encoding", str(workload.pos_encoding)),
            ("deterministic", str(workload.deterministic)),
        )
        signature = InputSignature(
            workload.op, dtypes=workload.dtypes, layouts=workload.layouts,
            specializations=specializations,
            inference=workload.inference,
        )
        hardware = HardwareSignature(
            soc_model=capability.soc_version, aicore_count=capability.ai_core_count,
        )
        # WorkloadSpec's flat static_dims/dynamic_bounds have no tensor/axis
        # mapping for ShapeRange. It also has no per-input physical_layouts;
        # logical layouts and QuantSpec's storage layout cannot supply ND/NZ.
        # Exclude only these tensor-only checks, preserving historical manifest
        # behavior. All shared CapabilitySpec checks use the existing predicate.
        # Manifest constraints additionally express whole dtype/layout tuples,
        # device features, quantization and determinism; keep them alongside it.
        return tuple(
            CompatibilityReport(
                d.kernel_id,
                d.constraints.unsupported_reasons(workload, capability)
                + replace(d.capability, shape_ranges=(), physical_layouts=())
                .unsupported_reasons(signature, hardware)
                + d.unsupported_target_reasons(capability.soc_version),
            )
            for d in self.descriptors(workload.op)
        )

    def select(self, workload, capability, *, backend=None,
               allow_reference=False, tuned_kernel_ids=()):
        """Select a manifest descriptor for the existing build_plan API.

        Tensor frontends continue to use runtime.dispatch; this path consumes
        WorkloadSpec/DeviceCapability, as required by manifest clients. Shared
        selection policies and T7 tiers match dispatch. With backend=None the
        global default applies; explicit 'auto' bypasses that default.

        tuned_kernel_ids is the manifest caller's ranking, applied only after
        filtering. query_tuning_record requires full InputSignature (including
        tensor shapes/physical layouts) and HardwareSignature (including UB);
        the partial view in explain() is not a valid database lookup key.
        This API returns a descriptor, without dispatch's explicit kernel_id
        override, fallback chain or decision cache. Every call re-reads policy.
        """
        from .dispatch import DispatchError, _sort_key

        if backend is None:
            backend = config.default_backend()
        policy = None if backend == "auto" else Backend(backend)
        order = config.dispatch_order()
        allow_custom = config.allow_custom_kernels()
        jit_disabled = config.jit_disabled()
        candidates = []
        rejected = []
        for report in self.explain(workload, capability):
            descriptor = self.get(report.kernel_id)
            reasons = list(report.reasons)
            if policy is not None and descriptor.backend != policy:
                reasons.append("forced backend does not match")
            elif (policy is None and descriptor.backend == Backend.CUSTOM
                  and not allow_custom):
                reasons.append(
                    "custom requires FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS=1 in auto"
                )
            if (descriptor.backend == Backend.REFERENCE and not allow_reference
                    and policy != Backend.REFERENCE):
                reasons.append("reference requires explicit opt-in")
            if descriptor.compile_mode == CompileMode.JIT and jit_disabled:
                reasons.append("JIT disabled (FLASHINFER_NPU_JIT_DISABLE=1)")
            if reasons:
                rejected.append(f"{descriptor.kernel_id}: {', '.join(reasons)}")
            else:
                candidates.append(descriptor)
        if not candidates:
            raise DispatchError("no compatible kernel; " + "; ".join(rejected))
        tuning = {kernel_id: i for i, kernel_id in enumerate(tuned_kernel_ids)}
        return min(candidates, key=lambda d: (
            tuning.get(d.kernel_id, len(tuning)),
            *_sort_key(d, order),
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
    scoped = _SCOPED_REGISTRY.get()
    if scoped is not None:
        return scoped
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_REGISTRY is None:
                from ..ascendc import ascendc_descriptors

                # Historical Python builtin declarations retain their own
                # catalog. Packaged production manifests join the same global
                # registry before it is frozen, with ordinary dispatch rules.
                registry = KernelRegistry(_builtin_descriptors() + ascendc_descriptors())
                registry.freeze()
                _GLOBAL_REGISTRY = registry
    return _GLOBAL_REGISTRY


@contextmanager
def kernel_registry_scope(registry: KernelRegistry):
    """Use a frozen experimental registry in this execution context only.

    Dispatch and runner lookup share the same registry. The process default is
    untouched; nested scopes restore their parent even when execution fails.
    """
    if not isinstance(registry, KernelRegistry) or not registry.frozen:
        raise ValueError("kernel_registry_scope requires a frozen KernelRegistry")
    token = _SCOPED_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _SCOPED_REGISTRY.reset(token)


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
