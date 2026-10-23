"""dispatcher：逐实现过滤 → 选择来源栈 → 选择栈内实现（T19A）。

要点：
- 来源栈由 registry 提供；栈边界取其最优可用实现的既有 T7 排序键，
  选栈后再在栈内按同一个键选实现，不增加无实测依据的排序规则；
- 显式 ``backend=`` 只缩小候选集，不改变选择算法（§5.3.1）；
- 显式 ``kernel_id=`` 直接锁定唯一实现，优先级最高（§3.2.2）；
- 排序先查 tuning 记录（进阶特性，当前恒未命中），冠军须通过上述过滤；
  未命中落可配置的确定性阶梯：external > CANN/ACLNN > 自研 AOT
  > 自研 JIT/未声明 AOT > reference。这是默认策略，不代表实测排名；
- 决策缓存（§5.3.2）：key 覆盖全部可影响决策的输入，含 registry
  注册计数——注册新 kernel 自动使旧决策缓存失效；
- 同一 workload 下选择结果确定可复现，全部过滤原因可查询（explain_dispatch）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple, Union

from .. import config
from ..tuning import query_tuning_record
from .hardware import HardwareSignature
from .registry import ImplementationStack, KernelRegistry, get_global_registry
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
class StackReport:
    """栈的资格、栈间比较与逐实现的栈内比较。

    accepted 表示至少一个实现通过过滤，selected 表示最终选中此栈。
    representative_kernel_id 是本栈的首选可用实现（未通过过滤时为 None）。
    implementations 保留每个实现的拒绝原因，并为通过者添加栈内排序原因。
    DispatchDecision.reports 仍只报告原有的逐实现过滤，保持调用者契约。
    """

    stack_id: str
    accepted: bool
    selected: bool = False
    representative_kernel_id: Optional[str] = None
    reasons: Tuple[str, ...] = ()
    implementations: Tuple[CandidateReport, ...] = ()


@dataclass(frozen=True)
class DispatchDecision:
    """dispatch 输出：选定 kernel、后备链、选型依据与完整决策链。"""

    kernel_id: str
    backend: Backend
    fallback_chain: Tuple[str, ...] = ()
    reason: str = ""
    reports: Tuple[CandidateReport, ...] = ()
    cache_hit: bool = False
    stack_id: str = ""
    stack_reports: Tuple[StackReport, ...] = ()


_TIER_NAMES = {
    "external": "已验证外部仓实现",
    "cann": "CANN / ACLNN 算子",
    "aot": "自研 AOT",
    "jit": "自研 JIT / 未声明 AOT",
    "reference": "reference",
}

# 每个 Backend 都须显式归类；新增 Backend 时缺项会报错，不静默兜底。
# None 仅用于明确列出的技术后端，由其 compile_mode 进一步分档。
_BACKEND_GROUPS = {
    Backend.CANN: "cann",
    Backend.ASCENDC: None,
    Backend.TRITON: None,
    Backend.EXTERNAL: "external",
    Backend.CUSTOM: None,
    Backend.REFERENCE: "reference",
    Backend.ASCENDC_AOT: "aot",
    Backend.ASCENDC_JIT: "jit",
    Backend.ACLNN: "cann",
}
_COMPILE_GROUPS = {
    CompileMode.AOT: "aot",
    CompileMode.JIT: "jit",
    CompileMode.NONE: "jit",
}


def _heuristic_group(descriptor: KernelDescriptor) -> str:
    group = _BACKEND_GROUPS[descriptor.backend]
    if group is None:
        return _COMPILE_GROUPS[descriptor.compile_mode]
    return group


def _heuristic_tier(
    descriptor: KernelDescriptor, order: Optional[Tuple[str, ...]] = None,
) -> int:
    """可配置的确定性启发式阶梯；历史 backend 按其声明的身份分档。"""
    if order is None:
        order = config.dispatch_order()
    return order.index(_heuristic_group(descriptor))


def _sort_key(descriptor: KernelDescriptor, order: Optional[Tuple[str, ...]] = None):
    # 同层 tie-break：priority 高者优先 → kernel_id 字典序兜底（全确定性）
    return (_heuristic_tier(descriptor, order), -descriptor.priority, descriptor.kernel_id)


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
    无可用 tuning 冠军时按 ``FLASHINFER_NPU_DISPATCH_ORDER`` 排序。
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

    # 每次读取并验证；快照同时用于缓存 key 与本次排序，避免配置串用。
    order = config.dispatch_order()
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
            order,
            registry.revision,
        )
        with _cache_lock:
            cached = _decision_cache.get(cache_key)
        if cached is not None:
            return replace(cached, cache_hit=True)

    decision = _dispatch_uncached(
        signature, hardware, policy, kernel_id, allow_reference, registry, order
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
    order: Tuple[str, ...],
) -> DispatchDecision:
    # 显式 kernel_id：直接锁定唯一实现，backend 随之确定（§5.1）。
    # capability 不满足时不静默换 kernel，抛错附原因（§12）。
    # 此覆盖分支按既有契约跳过 auto opt-in/JIT/排序；manifest select()
    # 没有 kernel_id 覆盖入口，backend= 仍走普通候选过滤。
    if kernel_id is not None:
        try:
            descriptor = registry.get(kernel_id)
        except KeyError as exc:
            raise DispatchError(str(exc)) from exc
        reasons = (
            descriptor.capability.unsupported_reasons(signature, hardware)
            + descriptor.unsupported_target_reasons(hardware.soc_model)
        )
        reports = (
            CandidateReport(kernel_id, descriptor.backend, not reasons, reasons),
        )
        stack_id = registry.stack_id_for(kernel_id)
        if reasons:
            raise DispatchError(
                "显式指定 kernel %s 不支持当前输入：%s"
                % (kernel_id, "；".join(reasons))
                + "\n  第一级：显式 kernel_id 锁定栈 %s；其他栈未评估" % stack_id
                + "\n  第二级：指定实现 %s 被拒绝：%s；其他配置未评估" % (kernel_id, "；".join(reasons))
            )
        return DispatchDecision(
            kernel_id=kernel_id,
            backend=descriptor.backend,
            reason="显式 kernel_id 锁定",
            reports=reports,
            stack_id=stack_id,
            stack_reports=(StackReport(
                stack_id, True, True, kernel_id,
                ("显式 kernel_id 锁定来源栈；其他栈未评估",),
                (replace(reports[0], reasons=(
                    "显式 kernel_id 锁定；capability / target 通过；其他配置未评估",
                )),),
            ),),
        )

    # 一个决策使用同一份登记快照。过滤过程中允许接入 external/custom，
    # 新记录留到下次 revision 决策，不能混入本次未评估的栈报告。
    stacks = registry.stacks(op=signature.op)
    descriptors = sorted(
        (d for stack in stacks for d in stack.descriptors), key=lambda d: d.kernel_id,
    )
    reports = []
    survivors = []
    allow_custom = config.allow_custom_kernels()
    for descriptor in descriptors:
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
        # 2/3. capability 过滤：语义（dtype/layout/特化参数/张量规模区间）
        #      + 二进制（SoC/AI Core 数），M0 统一由 CapabilitySpec 承担
        # manifest select() 另保留 KernelConstraints：整组 dtype/layout、
        # 设备 features/supported_dtypes、QuantSpec 与确定性要求。这里的
        # InputSignature/HardwareSignature 无完整对应字段，不反推 WorkloadSpec。
        reasons.extend(
            descriptor.capability.unsupported_reasons(signature, hardware)
        )
        reasons.extend(descriptor.unsupported_target_reasons(hardware.soc_model))
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

    eligible = {}
    for descriptor in survivors:
        eligible.setdefault(registry.stack_id_for(descriptor.kernel_id), []).append(descriptor)

    if not survivors:
        raise DispatchError(
            _format_no_candidate(signature, policy, _stack_reports(
                stacks, tuple(reports), eligible, None, None, order,
            ))
        )

    # 4. 查询只发生在 capability/policy 过滤之后，记录不能复活已淘汰的实现。
    #    tuning 仅给冠军 ID，其余后备仍按同一确定性启发式排序。
    tuned_kernel_id = query_tuning_record(signature, hardware)
    tuned = next((d for d in survivors if d.kernel_id == tuned_kernel_id), None)

    # 第一级：一个来源只有在至少一个实现可用时才可选。栈的比较边界取
    # 最优可用实现的 T7 键；不能用 Backend/静态栈 priority 替代，否则
    # 同栈含 AOT/JIT 或部分配置被淘汰时会改变 T6/T7 的既有结果。
    frontiers = {
        stack_id: min(members, key=lambda d: _sort_key(d, order))
        for stack_id, members in eligible.items()
    }
    selected_stack = (
        registry.stack_id_for(tuned.kernel_id) if tuned is not None else
        min(frontiers, key=lambda stack_id: _sort_key(frontiers[stack_id], order))
    )

    # 第二级：只在选中栈的可用实现中选择。T19A 复用现有的确定性规则，
    # 不引入自动调优、性能采样或配置搜索协议。
    local_order = sorted(eligible[selected_stack], key=lambda d: _sort_key(d, order))
    champion = tuned if tuned is not None else local_order[0]
    if tuned is not None:
        reason = "实测 tuning 记录命中：kernel_id=%s" % tuned_kernel_id
    else:
        tier = _heuristic_tier(champion, order)
        reason = "确定性启发式 tier %d（%s）" % (
            tier, _TIER_NAMES[_heuristic_group(champion)],
        )
        if tuned_kernel_id is not None:
            reason += "；tuning 记录 %s 不在当前可用候选中" % tuned_kernel_id

    # 后备是既有的逐 kernel 顺序，不能因分栈而把同栈后备强行提前。
    ordered = sorted(survivors, key=lambda d: _sort_key(d, order))
    return DispatchDecision(
        kernel_id=champion.kernel_id,
        backend=champion.backend,
        fallback_chain=tuple(d.kernel_id for d in ordered if d.kernel_id != champion.kernel_id),
        reason=reason,
        reports=tuple(reports),
        stack_id=selected_stack,
        stack_reports=_stack_reports(
            stacks, tuple(reports), eligible, selected_stack,
            tuned.kernel_id if tuned is not None else None, order,
        ),
    )


def _rank_description(descriptor: KernelDescriptor, order: Tuple[str, ...]) -> str:
    return "tier=%d（%s）, priority=%d, kernel_id=%s" % (
        _heuristic_tier(descriptor, order), _TIER_NAMES[_heuristic_group(descriptor)],
        descriptor.priority, descriptor.kernel_id,
    )


def _rank_comparison(winner, other, order) -> str:
    """指出既有 T7 键中首个决定胜负的分量，不把通过过滤等同于选中。"""
    a, b = _sort_key(winner, order), _sort_key(other, order)
    if a[0] != b[0]:
        return "tier %d < %d" % (a[0], b[0])
    if a[1] != b[1]:
        return "同 tier，priority %d > %d" % (winner.priority, other.priority)
    return "同 tier / priority，kernel_id 字典序 %s < %s" % (a[2], b[2])


def _stack_reports(
    stacks: Tuple[ImplementationStack, ...],
    reports: Tuple[CandidateReport, ...],
    eligible,
    selected_stack: Optional[str],
    tuned_kernel_id: Optional[str],
    order: Tuple[str, ...],
) -> Tuple[StackReport, ...]:
    filters = {report.kernel_id: report for report in reports}
    local_first = {}
    for stack_id, members in eligible.items():
        local_first[stack_id] = next(
            (d for d in members if d.kernel_id == tuned_kernel_id),
            min(members, key=lambda d: _sort_key(d, order)),
        )
    winner = local_first.get(selected_stack)
    result = []
    for stack in stacks:
        first = local_first.get(stack.stack_id)
        implementations = []
        for descriptor in stack.descriptors:
            report = filters[descriptor.kernel_id]
            if report.accepted:
                rank = _rank_description(descriptor, order)
                if descriptor.kernel_id == first.kernel_id:
                    detail = (
                        "栈内实测 tuning 记录命中" if first.kernel_id == tuned_kernel_id else
                        "栈内唯一通过过滤的实现" if len(eligible[stack.stack_id]) == 1 else
                        "栈内 T7 排序第一"
                    )
                elif first.kernel_id == tuned_kernel_id:
                    detail = "栈内让位于通过过滤的 tuning 冠军 %s" % first.kernel_id
                else:
                    detail = "栈内排序落后于 %s：%s" % (
                        first.kernel_id, _rank_comparison(first, descriptor, order),
                    )
                report = replace(report, reasons=("通过全部候选过滤", rank, detail))
            implementations.append(report)
        if first is None:
            reasons = ("栈不可用：栈内全部 %d 个实现被过滤" % len(implementations),) + tuple(
                "%s：%s" % (report.kernel_id, "；".join(report.reasons))
                for report in implementations
            )
        else:
            reasons = ("栈内 %d/%d 个实现通过过滤；比较边界：%s" % (
                len(eligible[stack.stack_id]), len(implementations),
                _rank_description(first, order),
            ),)
            if stack.stack_id == selected_stack:
                reasons += (("通过过滤的 tuning 冠军锁定此栈" if tuned_kernel_id else
                             "栈间 T7 排序第一"),)
            elif tuned_kernel_id:
                reasons += ("栈可用但未选中：让位于 tuning 冠军所在栈 %s" % selected_stack,)
            else:
                reasons += ("栈可用但未选中：落后于 %s：%s" % (
                    selected_stack, _rank_comparison(winner, first, order),
                ),)
        result.append(StackReport(
            stack.stack_id, first is not None, stack.stack_id == selected_stack,
            first.kernel_id if first is not None else None, reasons, tuple(implementations),
        ))
    return tuple(result)


def _format_stack_reports(reports: Tuple[StackReport, ...]):
    lines = ["  第一级：选择实现栈"]
    for stack in reports:
        mark = "select" if stack.selected else "accept" if stack.accepted else "reject"
        lines.append("    [%s] %s：%s" % (mark, stack.stack_id, "；".join(stack.reasons)))
    lines.append("  第二级：各栈内的实现（含逐条过滤与排序原因）")
    for stack in reports:
        lines.append("    栈 %s，首选=%s%s" % (
            stack.stack_id, stack.representative_kernel_id or "无",
            "（最终选中栈）" if stack.selected else "",
        ))
        for report in stack.implementations:
            # 保留原 [accept]/[reject] 过滤标记；首选关系由上方字段与
            # 逐实现排序理由表达，不能把 accept 悄悄改成 selected。
            mark = "accept" if report.accepted else "reject"
            lines.append("      [%s] %s (%s)：%s" % (
                mark, report.kernel_id, report.backend.value, "；".join(report.reasons),
            ))
    return lines


def _format_no_candidate(
    signature: InputSignature,
    policy: Union[str, Backend],
    reports: Tuple[StackReport, ...],
) -> str:
    lines = [
        "无可用 kernel：op=%s，backend policy=%s。逐条淘汰原因："
        % (signature.op, policy if policy == "auto" else policy.value)
    ]
    lines.extend(_format_stack_reports(reports))
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
    lines.extend(_format_stack_reports(decision.stack_reports))
    if decision.fallback_chain:
        lines.append("  fallback：%s" % " → ".join(decision.fallback_chain))
    return "\n".join(lines)


def clear_dispatch_cache() -> None:
    """清空决策缓存。一般无需调用——registry.revision 已保证正确性。"""
    with _cache_lock:
        _decision_cache.clear()
