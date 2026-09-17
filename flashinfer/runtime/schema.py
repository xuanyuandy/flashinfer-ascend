"""runtime 静态契约：Backend / CapabilitySpec / KernelDescriptor（§5、§11）。

设计要点（§5.1）：调度涉及 op / backend / kernel 三个粒度递进的概念——
- op      语义层（"做什么"），如 ``attention.decode``
- backend 实现来源层，是 kernel 的分类属性，不是独立决策对象
- kernel  唯一可执行单元，调度决策只在 kernel 粒度上发生
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

from .hardware import HardwareSignature
from .signature import InputSignature


class Backend(str, Enum):
    """kernel 的实现来源分类（§5.2）。

    ``custom`` 标记的是来源与信任等级而非实现技术——用户用 Ascend C 写的
    kernel 也归类为 custom，调度策略关心"是否经过质量门禁"（§5.2）。
    """

    CANN = "cann"
    ASCENDC = "ascendc"
    TRITON = "triton"
    CUSTOM = "custom"
    REFERENCE = "reference"


class CompileMode(str, Enum):
    """自研 kernel 的编译方式（§7）；CANN / reference 取 ``NONE``。"""

    AOT = "aot"
    JIT = "jit"
    NONE = "none"


class DispatchError(RuntimeError):
    """无 kernel 满足输入签名与调度策略时抛出（§12），附逐条淘汰原因。"""


@dataclass(frozen=True)
class CapabilitySpec:
    """kernel 的能力声明：能对哪些 (输入签名 × 硬件签名) 执行。

    空集合表示"不限制"。``specializations`` 逐项给出允许值集合——
    签名中出现的键若声明了允许值则必须命中，未声明的键不受限。
    """

    op: str
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    soc_models: Tuple[str, ...] = ()
    min_aicore: int = 0
    specializations: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def unsupported_reasons(
        self, signature: InputSignature, hardware: HardwareSignature
    ) -> Tuple[str, ...]:
        """返回逐条不支持原因；空元组表示 capability 通过。"""
        reasons = []
        if signature.op != self.op:
            return (
                "op 不匹配：kernel 支持 %s，请求 %s" % (self.op, signature.op),
            )
        if self.dtypes:
            for dtype in signature.dtypes:
                if dtype and dtype not in self.dtypes:
                    reasons.append(
                        "dtype %s 不在支持集 %s" % (dtype, sorted(self.dtypes))
                    )
        if self.layouts:
            for layout in signature.layouts:
                if layout and layout not in self.layouts:
                    reasons.append(
                        "layout %s 不在支持集 %s" % (layout, sorted(self.layouts))
                    )
        if self.soc_models and hardware.soc_model not in self.soc_models:
            reasons.append(
                "SoC %s 不在支持集 %s"
                % (hardware.soc_model, sorted(self.soc_models))
            )
        if self.min_aicore > 0 and hardware.aicore_count < self.min_aicore:
            reasons.append(
                "AI Core 数 %d 低于要求 %d"
                % (hardware.aicore_count, self.min_aicore)
            )
        declared = dict(self.specializations)
        for key, value in signature.specializations:
            allowed = declared.get(key)
            if allowed is not None and value not in allowed:
                reasons.append(
                    "特化参数 %s=%s 不在允许集 %s" % (key, value, sorted(allowed))
                )
        return tuple(reasons)


# kernel 的可执行入口：reference 为 Python 可调用；自研 kernel 由
# AOT 产物加载 / JIT 编译流程填充（M0 阶段为 None，拉起时报错）。
KernelRunner = Callable[..., Any]


@dataclass(frozen=True)
class KernelDescriptor:
    """registry 条目：一个可执行 kernel 的全部静态描述（§11）。

    ``priority`` 用于确定性启发式的同层 tie-break（维护者声明，
    新版本号优先，§5.4.3）。
    """

    kernel_id: str
    backend: Backend
    capability: CapabilitySpec
    compile_mode: CompileMode = CompileMode.NONE
    priority: int = 0
    runner: Optional[KernelRunner] = None

    @property
    def op(self) -> str:
        return self.capability.op
