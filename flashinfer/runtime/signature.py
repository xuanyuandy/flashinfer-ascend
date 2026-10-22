"""输入签名：dispatch 与 JIT cache key 的输入侧（设计文档 §11）。

签名一旦构建即不可变、可哈希。dispatch 决策缓存（§5.3.2）与
JIT cache key（§8.1）都以它为输入侧。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class InputSignature:
    """一次算子调用的输入侧特征快照。

    字段取舍（§8.1）：
    - ``op``              语义算子名（如 ``attention.decode``），与实现无关
    - ``dtypes``          各张量输入的 dtype 名，按参数顺序
    - ``shapes``          各张量输入的完整 shape
    - ``layouts``         各张量输入的 layout 名（无 layout 语义的输入填空串）
    - ``specializations`` 非张量特化参数（causal、量化格式等），排序后的键值对
    - ``physical_layouts`` 各输入的 NPU 物理排布 ND / NZ；空串表示未知，
      空元组表示全部未知。其他 CANN 格式保持未知，不猜作 ND 或 NZ
    """

    op: str
    dtypes: Tuple[str, ...] = ()
    shapes: Tuple[Tuple[int, ...], ...] = ()
    layouts: Tuple[str, ...] = ()
    specializations: Tuple[Tuple[str, str], ...] = ()
    physical_layouts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            physical_layouts = tuple(self.physical_layouts)
        except TypeError as exc:
            raise ValueError("physical_layouts must contain ND / NZ / empty string values") from exc
        if isinstance(self.physical_layouts, str) or any(
            not isinstance(layout, str) or layout not in ("", "ND", "NZ")
            for layout in physical_layouts
        ):
            raise ValueError("physical_layouts must contain ND / NZ / empty string values")
        object.__setattr__(
            self, "physical_layouts", physical_layouts if any(physical_layouts) else (),
        )
        object.__setattr__(self, "dtypes", tuple(str(d) for d in self.dtypes))
        object.__setattr__(
            self, "shapes", tuple(tuple(int(x) for x in s) for s in self.shapes)
        )
        object.__setattr__(self, "layouts", tuple(str(v) for v in self.layouts))
        object.__setattr__(
            self,
            "specializations",
            tuple(sorted((str(k), str(v)) for k, v in self.specializations)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """稳定序列化形式（cache key 的 JSON payload 来源）。"""
        value = {
            "op": self.op,
            "dtypes": list(self.dtypes),
            "shapes": [list(s) for s in self.shapes],
            "layouts": list(self.layouts),
            "specializations": [list(kv) for kv in self.specializations],
        }
        # 未知排布是默认值，不改变历史签名的序列化形式。
        if self.physical_layouts:
            value["physical_layouts"] = list(self.physical_layouts)
        return value


def tensor_spec(tensor: Any) -> Tuple[str, Tuple[int, ...]]:
    """从张量对象提取 ``(dtype 名, shape)``。

    只依赖 ``.dtype`` / ``.shape`` 属性协议，不 import torch——
    host 单测（§14.1）可用轻量假对象。
    """
    dtype = str(getattr(tensor, "dtype", "unknown"))
    if dtype.startswith("torch."):
        dtype = dtype[len("torch."):]
    shape = tuple(int(d) for d in getattr(tensor, "shape", ()))
    return dtype, shape


def _tensor_physical_layout(tensor: Any) -> str:
    """只查询 NPU 张量元数据；host / 假张量不依赖 torch_npu。

    实测 torch_npu.get_npu_format 返回 Format(IntEnum)，旧版可能返回 int；
    ND=2，FRACTAL_NZ=29。其他格式（包括 NZ 的 C0 变体）不合并。
    """
    if getattr(getattr(tensor, "device", None), "type", None) not in (
        "npu", "privateuseone",
    ):
        return ""
    try:
        import torch_npu

        return {2: "ND", 29: "NZ"}.get(torch_npu.get_npu_format(tensor), "")
    except Exception:
        return ""


def build_signature(
    op: str,
    tensors: Iterable[Any],
    layouts: Optional[Iterable[str]] = None,
    specializations: Optional[Dict[str, Any]] = None,
) -> InputSignature:
    """从调用实参构建签名。``layouts`` 与张量一一对应，缺省为空串。

    物理排布从 NPU 张量查询，查询失败时未知；不从逻辑 layout / stride 推测，
    不执行格式转换。手工构造 InputSignature 的调用方需自行提供物理排布。
    """
    tensors = tuple(tensors)
    specs = [tensor_spec(t) for t in tensors]
    layout_tuple = tuple(layouts) if layouts is not None else ("",) * len(tensors)
    if len(layout_tuple) != len(tensors):
        raise ValueError("layouts 数量与张量数量不一致")
    return InputSignature(
        op=op,
        dtypes=tuple(s[0] for s in specs),
        shapes=tuple(s[1] for s in specs),
        layouts=layout_tuple,
        physical_layouts=tuple(_tensor_physical_layout(t) for t in tensors),
        specializations=tuple(
            sorted((str(k), str(v)) for k, v in (specializations or {}).items())
        ),
    )
