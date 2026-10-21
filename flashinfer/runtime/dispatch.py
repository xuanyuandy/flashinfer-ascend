"""dispatcher：backend policy 过滤 → capability 过滤 → 排序（设计文档 §5.3）。

要点：
- 决策单元是 kernel 不是 backend；backend 是选定 kernel 后附带确定的属性；
- 显式 ``backend=`` 只缩小候选集，不改变选择算法（§5.3.1）；
- 显式 ``kernel_id=`` 直接锁定唯一实现，优先级最高（§3.2.2）；
- 排序先查 tuning 记录（进阶特性，当前恒未命中），未命中落确定性
  启发式阶梯（§5.4.3）：自研 AOT > CANN > 自研 JIT > reference；
- 决策缓存（§5.3.2）：key 覆盖全部可影响决策的输入，含 registry
  注册计数——注册新 kernel 自动使旧决策缓存失效；
- 同一 workload 下选择结果确定可复现，全部过滤原因可查询（explain_dispatch）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple, Union

from .. import config
from .hardware import HardwareSignature
from .registry import KernelRegistry, get_global_registry
from .schema import Backend, CompileMode, DispatchError, KernelDescriptor
from .signature import InputSignature


@dataclass(frozen=True)
class CandidateReport:
    """单个候选的过滤结论与逐条原因（§5.3 第 5 步，explain 的数据源）。"""

    kernel_id: str
    backend: Backend
    accepted: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DispatchDecision:
    """dispatch 输出：选定 kernel、后备链、选型依据与完整决策链。"""

    kernel_id: str
    backend: Backend
    fallback_chain: Tuple[str, ...] = ()
    reason: str = ""
    reports: Tuple[CandidateReport, ...] = ()
    cache_hit: bool = False


_TIER_NAMES = {
    0: "自研 AOT",
    1: "CANN 算子",
    2: "自研 JIT",
    3: "reference",
}


def _heuristic_tier(descriptor: KernelDescriptor) -> int:
    """确定性启发式阶梯（§5.4.3）。"""
    if descriptor.backend == Backend.REFERENCE:
        return 3
    if descriptor.backend == Backend.CANN:
        return 1
    # 自研（ascendc / triton / custom）：AOT 产物匹配优先于 JIT
    return 0 if descriptor.compile_mode == CompileMode.AOT else 2


def _sort_key(descriptor: KernelDescriptor):
    # 同层 tie-break：priority 高者优先 → kernel_id 字典序兜底（全确定性）
    return (_heuristic_tier(descriptor), -descriptor.priority, descriptor.kernel_id)


_decision_cache: Dict[tuple, DispatchDecision] = {}
_cache_lock = threading.Lock()


def dispatch(
    signature: InputSignature,
    hardware: HardwareSignature,
    backend: Union[str, Backend, None] = "auto",
    kernel_id: Optional[str] = None,
    allow_reference: bool = False,
    registry: Optional[KernelRegistry] = None,
) -> DispatchDecision:
    """按 §5.3 算法选择 kernel。结果确定、可复现、可解释。

    ``backend=None`` 时取全局默认 ``FLASHINFER_NPU_BACKEND``（§13）。
    """
    registry = registry or get_global_registry()
    if backend is None:
        backend = config.default_backend()
    try:
        policy: Union[str, Backend] = (
            "auto" if backend == "auto" else Backend(backend)
        )
    except (ValueError, TypeError) as error:
        supported = ", ".join(("auto", *sorted(item.value for item in Backend)))
        cuda_backends = {"fa2", "fa3", "cudnn", "trtllm-gen", "cutlass", "cute-dsl", "cutile"}
        hint = (
            "这是 CUDA 专有后端；昇腾请使用 backend='auto' 或 backend='cann'。"
            if isinstance(backend, str) and backend in cuda_backends
            else "请使用本仓已登记的 backend 名称。"
        )
        raise DispatchError(
            f"不支持的 backend {backend!r}。{hint}支持的 backend 值：{supported}"
        ) from error

    cache_key = None
    if not config.dispatch_cache_disabled():
        cache_key = (
            signature,
            hardware,
            str(policy),
            kernel_id,
            allow_reference,
            config.allow_custom_kernels(),
            config.jit_disabled(),
            registry.revision,
        )
        with _cache_lock:
            cached = _decision_cache.get(cache_key)
        if cached is not None:
            return replace(cached, cache_hit=True)

    decision = _dispatch_uncached(
        signature, hardware, policy, kernel_id, allow_reference, registry
    )
    if cache_key is not None:
        with _cache_lock:
            _decision_cache[cache_key] = decision
    return decision


def _dispatch_uncached(
    signature: InputSignature,
    hardware: HardwareSignature,
    policy: Union[str, Backend],
    kernel_id: Optional[str],
    allow_reference: bool,
    registry: KernelRegistry,
) -> DispatchDecision:
    # 显式 kernel_id：直接锁定唯一实现，backend 随之确定（§5.1）。
    # capability 不满足时不静默换 kernel，抛错附原因（§12）。
    if kernel_id is not None:
        try:
            descriptor = registry.get(kernel_id)
        except KeyError as exc:
            raise DispatchError(str(exc)) from exc
        reasons = descriptor.capability.unsupported_reasons(signature, hardware)
        reports = (
            CandidateReport(kernel_id, descriptor.backend, not reasons, reasons),
        )
        if reasons:
            raise DispatchError(
                "显式指定 kernel %s 不支持当前输入：%s"
                % (kernel_id, "；".join(reasons))
            )
        return DispatchDecision(
            kernel_id=kernel_id,
            backend=descriptor.backend,
            reason="显式 kernel_id 锁定",
            reports=reports,
        )

    reports = []
    survivors = []
    allow_custom = config.allow_custom_kernels()
    for descriptor in registry.descriptors(op=signature.op):
        reasons = []
        # 1. backend policy 过滤（§5.3 第 1 步：policy 只是候选集过滤器）
        if policy != "auto" and descriptor.backend != policy:
            reasons.append("backend policy %s 排除" % policy.value)
        elif (
            policy == "auto"
            and descriptor.backend == Backend.CUSTOM
            and not allow_custom
        ):
            reasons.append(
                "自定义 kernel 默认不进 auto 候选池"
                "（需 FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS，§10.5.2）"
            )
        # reference 拒绝进热路径（§12）：显式 backend='reference' 或
        # allow_reference=True 才放行
        if (
            descriptor.backend == Backend.REFERENCE
            and not allow_reference
            and policy != Backend.REFERENCE
        ):
            reasons.append(
                "reference 需显式 opt-in（backend='reference' 或 allow_reference=True）"
            )
        # 2/3. capability 过滤：语义（dtype/layout/特化参数）
        #      + 二进制（SoC/AI Core 数），M0 统一由 CapabilitySpec 承担
        reasons.extend(
            descriptor.capability.unsupported_reasons(signature, hardware)
        )
        # JIT 全局禁用（§13）：仅 AOT / CANN 可用
        if (
            descriptor.compile_mode == CompileMode.JIT
            and config.jit_disabled()
        ):
            reasons.append("JIT 已禁用（FLASHINFER_NPU_JIT_DISABLE=1，§13）")
        accepted = not reasons
        reports.append(
            CandidateReport(
                descriptor.kernel_id, descriptor.backend, accepted, tuple(reasons)
            )
        )
        if accepted:
            survivors.append(descriptor)

    if not survivors:
        raise DispatchError(
            _format_no_candidate(signature, policy, tuple(reports))
        )

    # 4. 排序：tuning 记录优先（进阶特性 §9，当前恒未命中）；
    #    无记录落确定性启发式（§5.4.3）
    ordered = sorted(survivors, key=_sort_key)
    champion = ordered[0]
    tier = _heuristic_tier(champion)
    return DispatchDecision(
        kernel_id=champion.kernel_id,
        backend=champion.backend,
        fallback_chain=tuple(d.kernel_id for d in ordered[1:]),
        reason="确定性启发式 tier %d（%s）" % (tier, _TIER_NAMES[tier]),
        reports=tuple(reports),
    )


def _format_no_candidate(
    signature: InputSignature,
    policy: Union[str, Backend],
    reports: Tuple[CandidateReport, ...],
) -> str:
    lines = [
        "无可用 kernel：op=%s，backend policy=%s。逐条淘汰原因："
        % (signature.op, policy if policy == "auto" else policy.value)
    ]
    for report in reports:
        lines.append(
            "  - %s [%s]：%s"
            % (
                report.kernel_id,
                report.backend.value,
                "；".join(report.reasons) or "（未评估）",
            )
        )
    if not reports:
        lines.append("  （registry 中无该 op 的任何 kernel）")
    return "\n".join(lines)


def explain_dispatch(
    signature: InputSignature,
    hardware: HardwareSignature,
    backend: Union[str, Backend, None] = "auto",
    kernel_id: Optional[str] = None,
    allow_reference: bool = False,
    registry: Optional[KernelRegistry] = None,
) -> str:
    """打印完整决策链（§5.4.3 可解释性要求的落地）。"""
    decision = dispatch(
        signature,
        hardware,
        backend=backend,
        kernel_id=kernel_id,
        allow_reference=allow_reference,
        registry=registry,
    )
    lines = [
        "dispatch %s → %s（backend=%s）"
        % (signature.op, decision.kernel_id, decision.backend.value),
        "  选型依据：%s" % decision.reason,
    ]
    for report in decision.reports:
        mark = "accept" if report.accepted else "reject"
        detail = "；".join(report.reasons) if report.reasons else "-"
        lines.append(
            "  [%s] %s (%s)：%s"
            % (mark, report.kernel_id, report.backend.value, detail)
        )
    if decision.fallback_chain:
        lines.append("  fallback：%s" % " → ".join(decision.fallback_chain))
    return "\n".join(lines)


def clear_dispatch_cache() -> None:
    """清空决策缓存。一般无需调用——registry.revision 已保证正确性。"""
    with _cache_lock:
        _decision_cache.clear()
