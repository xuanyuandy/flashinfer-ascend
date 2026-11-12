"""runtime 静态契约：Backend / CapabilitySpec / KernelDescriptor（§5、§11）。

设计要点（§5.1，T19A）：调度涉及 op / 实现栈 / kernel——
- op      语义层（"做什么"），如 ``attention.decode``
- stack   registry 按来源分组的视图，外部仓的命名空间参与身份
- backend 来源分类属性；多个栈可同属 EXTERNAL，Ascend C AOT/JIT 可同栈
- kernel  栈内唯一可执行单元；既有 capability 与 T7 排序不变
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

from .features import INFERENCE_FEATURE_ALIASES, validate_feature
from .hardware import HardwareSignature
if TYPE_CHECKING:
    from .signature import InputSignature


class Backend(str, Enum):
    """kernel 的实现来源分类（§5.2）。

    ``custom`` 标记的是来源与信任等级而非实现技术——用户用 Ascend C 写的
    kernel 也归类为 custom，调度策略关心"是否经过质量门禁"（§5.2）。
    ``external`` 表示本仓主动对接并验证的外部仓实现，默认参与 auto。
    """

    CANN = "cann"
    ASCENDC = "ascendc"
    TRITON = "triton"
    EXTERNAL = "external"
    CUSTOM = "custom"
    REFERENCE = "reference"
    # Historical artifact/provider contracts retain their serialized identities.
    ASCENDC_AOT = "ascendc_aot"
    ASCENDC_JIT = "ascendc_jit"
    ACLNN = "aclnn"


class CompileMode(str, Enum):
    """自研 kernel 的编译方式（§7）；CANN / reference 取 ``NONE``。"""

    AOT = "aot"
    JIT = "jit"
    NONE = "none"


class GraphCapability(str, Enum):
    """kernel 的入图性质声明，不参与 eager capability 过滤或排序。"""

    CAPTURE_SAFE = "capture_safe"
    NEEDS_PARAM_UPDATE = "needs_param_update"
    NOT_CAPTURABLE = "not_capturable"


class DispatchError(RuntimeError):
    """无 kernel 满足输入签名与调度策略时抛出（§12），附逐条淘汰原因。"""


class SchemaError(ValueError):
    """A versioned runtime contract failed validation."""


class ArtifactVerificationError(SchemaError):
    """Artifact bytes or their resolved location do not match provenance."""


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _hash_field(name, value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SchemaError(f"{name} must be lowercase SHA-256")


class _Record:
    def to_dict(self):
        return _json_value(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {f.name for f in fields(cls)}:
            raise SchemaError(f"{cls.__name__} fields do not match schema")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc

    @property
    def fingerprint(self):
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class QuantSpec(_Record):
    scheme: str
    storage_dtype: str
    compute_dtype: str
    accumulator_dtype: str
    scale_dtype: str = "float32"
    granularity: str = "tensor"
    group_size: Optional[Tuple[int, ...]] = None
    axis: Optional[Tuple[int, ...]] = None
    has_zero_point: bool = False
    physical_layout: str = "logical"
    packing_order: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise SchemaError("unsupported quantization schema version")
        if self.scheme not in {"symmetric", "asymmetric"}:
            raise SchemaError("unsupported quantization scheme")
        if self.scheme == "asymmetric" and not self.has_zero_point:
            raise SchemaError("asymmetric quantization requires zero point")
        if self.storage_dtype.endswith("4_packed") and self.packing_order is None:
            raise SchemaError("packed int4 requires packing_order")
        if self.packing_order not in (None, "low_nibble_first", "high_nibble_first"):
            raise SchemaError("unsupported packing_order")
        if self.granularity not in {"tensor", "channel", "token", "group", "block"}:
            raise SchemaError("unsupported quantization granularity")
        if self.granularity in {"group", "block"} and not self.group_size:
            raise SchemaError(f"{self.granularity} quantization requires group_size")
        for name in ("axis", "group_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(int(x) for x in value))
        if self.group_size is not None and any(x <= 0 for x in self.group_size):
            raise SchemaError("group_size must be positive")


def _feature_strings(name, values):
    if isinstance(values, (str, bytes, Mapping)):
        raise SchemaError(f"{name} must be a sequence of non-empty strings")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise SchemaError(f"{name} must be a sequence") from exc
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise SchemaError(f"{name} must contain non-empty strings")
    return values


def _quantization_bindings(values):
    try:
        pairs = tuple(tuple(v) for v in values)
    except TypeError as exc:
        raise SchemaError("quantization must contain (tensor_index, QuantSpec) pairs") from exc
    if any(len(v) != 2 or type(v[0]) is not int or v[0] < 0
           or not isinstance(v[1], QuantSpec) for v in pairs):
        raise SchemaError("quantization must contain non-negative tensor_index / QuantSpec pairs")
    if len({v[0] for v in pairs}) != len(pairs):
        raise SchemaError("quantization must not repeat tensor_index")
    return tuple(sorted(pairs, key=lambda v: v[0]))


def _decode_quantization(values):
    try:
        return tuple((index, QuantSpec.from_dict(spec)) for index, spec in values)
    except (TypeError, ValueError, AttributeError) as exc:
        raise SchemaError("invalid quantization bindings") from exc


def _functional_values(values):
    """Canonicalize explicit scalar facts without guessing missing features."""
    try:
        pairs = tuple(tuple(v) for v in (values.items() if isinstance(values, Mapping) else values))
        if any(len(v) != 2 for v in pairs):
            raise ValueError("features require (name, value) pairs")
        for name, value in pairs:
            validate_feature(name, value)
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError("features must not repeat a name")
        return tuple(sorted(pairs))
    except (TypeError, ValueError) as error:
        raise SchemaError(str(error)) from error


@dataclass(frozen=True)
class FeatureConstraint(_Record):
    """An explicit, typed functional choice; missing call metadata rejects."""

    name: str
    allowed_values: Tuple[Union[str, bool, int], ...]

    def __post_init__(self):
        try:
            if isinstance(self.allowed_values, (str, bytes, Mapping)):
                raise ValueError("allowed_values must be a non-empty sequence")
            values = tuple(self.allowed_values)
            if not values:
                raise ValueError("allowed_values must be a non-empty sequence")
            for value in values:
                validate_feature(self.name, value)
            object.__setattr__(self, "allowed_values", tuple(sorted(set(values))))
        except (TypeError, ValueError) as error:
            raise SchemaError(str(error)) from error


@dataclass(frozen=True)
class InferenceSpec(_Record):
    """Explicit call semantics; never infer quantization or features from dtype.

    Tensor indices follow InputSignature, including prospective outputs.
    QuantSpec retains scales, grouping, zero-point and packing per operand.
    Empty strings / None mean unavailable metadata, not a supported feature.
    New scalar functions use ``features`` and the typed feature catalogue;
    flat scalar fields remain compatibility aliases for the same matcher.
    """

    compute_dtype: str = ""
    accumulator_dtype: str = ""
    quantization: Tuple[Tuple[int, QuantSpec], ...] = ()
    attention_mode: str = ""
    attention_variant: str = ""
    kv_cache_type: str = ""
    mask_type: str = ""
    pos_encoding: str = ""
    causal: Optional[bool] = None
    sliding_window: Optional[bool] = None
    logits_soft_cap: Optional[float] = None
    fusion_ops: Tuple[str, ...] = ()
    auxiliary_outputs: Tuple[str, ...] = ()
    sparse_format: str = ""
    routing_method: str = ""
    moe_top_k: Optional[int] = None
    expert_parallel_size: Optional[int] = None
    deterministic: bool = False
    features: Tuple[Tuple[str, Union[str, bool, int]], ...] = ()

    def __post_init__(self):
        for name in ("compute_dtype", "accumulator_dtype", "attention_mode", "attention_variant",
                     "kv_cache_type", "mask_type", "pos_encoding", "sparse_format", "routing_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise SchemaError(f"{name} must be a string; empty means unknown")
        for name in ("causal", "sliding_window"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise SchemaError(f"{name} must be bool or None")
        if type(self.deterministic) is not bool:
            raise SchemaError("deterministic must be bool")
        for name in ("moe_top_k", "expert_parallel_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise SchemaError(f"{name} must be a positive integer or None")
        if self.logits_soft_cap is not None:
            if (type(self.logits_soft_cap) not in (int, float)
                    or not math.isfinite(self.logits_soft_cap) or self.logits_soft_cap < 0):
                raise SchemaError("logits_soft_cap must be finite and non-negative or None")
            object.__setattr__(self, "logits_soft_cap", float(self.logits_soft_cap))
        object.__setattr__(self, "quantization", _quantization_bindings(self.quantization))
        for name in ("fusion_ops", "auxiliary_outputs"):
            object.__setattr__(self, name, _feature_strings(name, getattr(self, name)))
        object.__setattr__(self, "features", _functional_values(self.features))
        self.feature_values()  # Reject contradictory flat/structured declarations.

    def feature_values(self):
        """One view of functional facts, including the flat compatibility fields."""
        values = dict(self.features)
        for field, key, _ in INFERENCE_FEATURE_ALIASES:
            value = getattr(self, field)
            if value is not None and value != "":
                if key in values and values[key] != value:
                    raise SchemaError(f"feature {key} contradicts {field}")
                values[key] = value
        return tuple(sorted(values.items()))

    def to_dict(self):
        return {f.name: _json_value(getattr(self, f.name)) for f in fields(self)
                if getattr(self, f.name) != f.default}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or not set(value) <= {f.name for f in fields(cls)}:
            raise SchemaError("InferenceSpec fields do not match schema")
        values = dict(value)
        values["quantization"] = _decode_quantization(values.get("quantization", ()))
        return cls(**values)


@dataclass(frozen=True)
class WorkloadSpec(_Record):
    op: str
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    static_dims: Tuple[int, ...] = ()
    dynamic_bounds: Tuple[int, ...] = ()
    attributes: Tuple[Tuple[str, str], ...] = ()
    quant_specs: Tuple[QuantSpec, ...] = ()
    causal: bool = False
    pos_encoding: str = "NONE"
    deterministic: bool = False
    schema_version: int = 1
    inference: Optional[InferenceSpec] = None

    def __post_init__(self):
        if self.schema_version != 1:
            raise SchemaError("unsupported workload schema version")
        if not self.op:
            raise SchemaError("workload op must be non-empty")
        for name in ("dtypes", "layouts", "static_dims", "dynamic_bounds", "quant_specs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "attributes", tuple(sorted(
            (str(k), str(v)) for k, v in self.attributes
        )))
        if self.inference is not None:
            value = self.inference
            if isinstance(value, Mapping):
                value = InferenceSpec.from_dict(value)
            if not isinstance(value, InferenceSpec):
                raise SchemaError("inference must be InferenceSpec or None")
            functional = dict(value.feature_values())
            for name, key in (("causal", "attention.causal"), ("pos_encoding", "position.encoding")):
                actual = functional.get(key)
                if actual not in (None, "") and actual != getattr(self, name):
                    raise SchemaError(f"inference {name} contradicts WorkloadSpec.{name}")
            if value.deterministic and not self.deterministic:
                raise SchemaError("inference deterministic requires WorkloadSpec.deterministic")
            if value.to_dict():
                bindings = tuple(dict.fromkeys(spec for _, spec in value.quantization))
                if self.quant_specs and set(self.quant_specs) != set(bindings):
                    raise SchemaError("inference quantization contradicts WorkloadSpec.quant_specs")
                if not self.quant_specs and bindings:
                    object.__setattr__(self, "quant_specs", bindings)
            object.__setattr__(self, "inference", value if value.to_dict() else None)

    def to_dict(self):
        value = super().to_dict()
        if self.inference is None:
            value.pop("inference")
        else:
            value["inference"] = self.inference.to_dict()
        return value

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["quant_specs"] = tuple(QuantSpec.from_dict(v) for v in values["quant_specs"])
        values.setdefault("inference", None)
        return super().from_dict(values)


@dataclass(frozen=True)
class DeviceCapability(_Record):
    soc_version: str
    soc_revision: str = ""
    ai_core_count: int = 0
    supported_dtypes: Tuple[str, ...] = ()
    features: Tuple[str, ...] = ()
    cann_version: str = ""
    torch_npu_version: str = ""
    compiler_version: str = ""

    def __post_init__(self):
        for name in ("supported_dtypes", "features"):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))


@dataclass(frozen=True)
class WorkspaceFormula(_Record):
    constant_bytes: int = 0
    dynamic_coefficients: Tuple[int, ...] = ()
    alignment: int = 32

    def __post_init__(self):
        object.__setattr__(self, "dynamic_coefficients", tuple(self.dynamic_coefficients))
        if self.constant_bytes < 0 or any(v < 0 for v in self.dynamic_coefficients):
            raise SchemaError("workspace coefficients must be non-negative")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise SchemaError("workspace alignment must be a power of two")

    def size_for(self, workload):
        if len(self.dynamic_coefficients) > len(workload.dynamic_bounds):
            raise SchemaError("workspace formula requires dynamic bounds")
        value = self.constant_bytes + sum(
            c * n for c, n in zip(self.dynamic_coefficients, workload.dynamic_bounds)
        )
        return (value + self.alignment - 1) // self.alignment * self.alignment


@dataclass(frozen=True)
class KernelConstraints(_Record):
    supported_socs: Tuple[str, ...] = ()
    dtype_signatures: Tuple[Tuple[str, ...], ...] = ()
    layout_signatures: Tuple[Tuple[str, ...], ...] = ()
    required_features: Tuple[str, ...] = ()
    quant_storage_dtypes: Tuple[str, ...] = ()
    deterministic: Optional[bool] = None

    def __post_init__(self):
        for name in ("supported_socs", "required_features", "quant_storage_dtypes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("dtype_signatures", "layout_signatures"):
            object.__setattr__(self, name, tuple(tuple(x) for x in getattr(self, name)))

    def unsupported_reasons(self, workload, capability):
        reasons = []
        if self.supported_socs and capability.soc_version not in self.supported_socs:
            reasons.append("unsupported SoC")
        if self.dtype_signatures and workload.dtypes not in self.dtype_signatures:
            reasons.append("unsupported dtype signature")
        if capability.supported_dtypes and any(
            d not in capability.supported_dtypes for d in workload.dtypes
        ):
            reasons.append("device does not support dtype")
        if self.layout_signatures and workload.layouts not in self.layout_signatures:
            reasons.append("unsupported layout signature")
        missing = set(self.required_features) - set(capability.features)
        if missing:
            reasons.append("missing features: " + ", ".join(sorted(missing)))
        if self.quant_storage_dtypes and (
            not workload.quant_specs or any(
                q.storage_dtype not in self.quant_storage_dtypes for q in workload.quant_specs
            )
        ):
            reasons.append("unsupported quant storage dtype")
        if workload.deterministic and self.deterministic is False:
            reasons.append("deterministic execution required")
        return tuple(reasons)


@dataclass(frozen=True)
class KernelCapabilityBinding(_Record):
    domain: str
    profile_id: str
    rule_id: str
    profile_fingerprint: str
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise SchemaError("unsupported capability binding schema version")
        if not all((self.domain, self.profile_id, self.rule_id)):
            raise SchemaError("capability binding identifiers must be non-empty")
        _hash_field("profile_fingerprint", self.profile_fingerprint)


class ArtifactKind(str, Enum):
    FILE = "file"
    JIT_SOURCE = "jit_source"
    BUILTIN = "builtin"


class ArtifactFormat(str, Enum):
    ASCENDC_OBJECT = "ascendc_object"
    ASCENDC_OP_PACKAGE = "ascendc_op_package"
    SHARED_LIBRARY = "shared_library"
    TRITON_NPUBIN = "triton_npubin"
    ASCENDC_SOURCE = "ascendc_source"
    ACLNN_BUILTIN = "aclnn_builtin"


@dataclass(frozen=True)
class ArtifactRef(_Record):
    """产物溯源；旧记录的入口仍可由 ``KernelLaunchABI.entry_point`` 提供。"""

    kind: ArtifactKind
    format: ArtifactFormat
    locator: str
    digest: str
    target_soc: str
    build_id: str
    size_bytes: Optional[int] = None
    schema_version: int = 1
    entry_point: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "kind", ArtifactKind(self.kind))
        object.__setattr__(self, "format", ArtifactFormat(self.format))
        if self.schema_version != 1:
            raise SchemaError("unsupported artifact schema_version")
        _hash_field("artifact digest", self.digest)
        if not self.target_soc or not self.build_id or not self.locator:
            raise SchemaError("artifact target_soc, build_id and locator required")
        if self.entry_point is not None and (
            not isinstance(self.entry_point, str) or not self.entry_point
            or any(c.isspace() or c == "\0" for c in self.entry_point)
        ):
            raise SchemaError("artifact entry_point must be a non-empty symbol")
        expected = {
            ArtifactFormat.ASCENDC_OBJECT: ArtifactKind.FILE,
            ArtifactFormat.ASCENDC_OP_PACKAGE: ArtifactKind.FILE,
            ArtifactFormat.SHARED_LIBRARY: ArtifactKind.FILE,
            ArtifactFormat.TRITON_NPUBIN: ArtifactKind.FILE,
            ArtifactFormat.ASCENDC_SOURCE: ArtifactKind.JIT_SOURCE,
            ArtifactFormat.ACLNN_BUILTIN: ArtifactKind.BUILTIN,
        }[self.format]
        if self.kind != expected:
            raise SchemaError("artifact kind does not match format")
        if self.kind == ArtifactKind.BUILTIN:
            if self.size_bytes is not None:
                raise SchemaError("builtin provider cannot declare size")
        else:
            p = PurePosixPath(self.locator)
            if (p.is_absolute() or "\\" in self.locator or ".." in p.parts
                    or str(p) != self.locator):
                raise SchemaError("artifact locator must be normalized relative POSIX path")
            if self.size_bytes is None or self.size_bytes < 0:
                raise SchemaError("artifact requires non-negative size")

    def to_dict(self):
        value = super().to_dict()
        # 保留 T1 恢复的旧记录序列化形式与指纹。
        if self.entry_point is None:
            value.pop("entry_point")
        return value

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values.setdefault("entry_point", None)
        return super().from_dict(values)

    def verify_bytes(self, payload):
        if self.kind == ArtifactKind.BUILTIN:
            raise ArtifactVerificationError("builtin provider has no file bytes")
        if len(payload) != self.size_bytes:
            raise ArtifactVerificationError("artifact size mismatch")
        if hashlib.sha256(payload).hexdigest() != self.digest:
            raise ArtifactVerificationError("artifact digest mismatch")

    def verify_file(self, package_root):
        root = Path(package_root).resolve()
        target = (root / self.locator).resolve()
        if not target.is_relative_to(root):
            raise ArtifactVerificationError("artifact escapes package root")
        try:
            self.verify_bytes(target.read_bytes())
        except OSError as exc:
            raise ArtifactVerificationError(str(exc)) from exc
        return target


@dataclass(frozen=True)
class SourceRef(_Record):
    """源码引用：语言、包内相对 POSIX 路径与文件内容的 SHA-256。

    语言使用非空标识字符串（如 ``ascendc`` / ``triton`` / ``c++``），
    不在静态 schema 中限制将来可接入的编译器。SOC 是 descriptor 的编译维度，
    同一份源码可以对应多个 SOC 的记录。构造引用不读取文件或触发编译。
    """

    language: str
    locator: str
    digest: str
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise SchemaError("unsupported source schema_version")
        if (not isinstance(self.language, str) or not self.language
                or any(c.isspace() or c == "\0" for c in self.language)):
            raise SchemaError("source language must be a non-empty identifier")
        _hash_field("source digest", self.digest)
        if not isinstance(self.locator, str) or not self.locator or "\0" in self.locator:
            raise SchemaError("source locator must be a normalized relative POSIX path")
        path = PurePosixPath(self.locator)
        if (path.is_absolute() or "\\" in self.locator or ".." in path.parts
                or self.locator == "." or str(path) != self.locator):
            raise SchemaError("source locator must be a normalized relative POSIX path")


@dataclass(frozen=True)
class KernelLaunchABI(_Record):
    abi_name: str
    entry_point: str
    argument_names: Tuple[str, ...]
    mutable_arguments: Tuple[str, ...] = ()
    stream_argument: str = "stream"
    pointer_width_bits: int = 64
    metadata_schema_version: int = 1
    schema_version: int = 1

    def __post_init__(self):
        for name in ("argument_names", "mutable_arguments"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(set(self.argument_names)) != len(self.argument_names):
            raise SchemaError("argument names must be unique")
        if not set(self.mutable_arguments) <= set(self.argument_names):
            raise SchemaError("mutable arguments must be an argument subset")
        if self.stream_argument not in self.argument_names:
            raise SchemaError("stream argument must occur in argument names")
        if self.pointer_width_bits not in (32, 64):
            raise SchemaError("pointer width must be 32 or 64")
        if self.schema_version != 1 or self.metadata_schema_version <= 0:
            raise SchemaError("unsupported launch ABI schema version")


class CPrimitive(str, Enum):
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"

    @property
    def struct_format(self):
        return {"u8": "B", "u16": "H", "u32": "I", "u64": "Q",
                "i8": "b", "i16": "h", "i32": "i", "i64": "q",
                "f32": "f", "f64": "d"}[self.value]

    @property
    def size_bytes(self):
        return struct.calcsize("<" + self.struct_format)


@dataclass(frozen=True)
class CFieldABI(_Record):
    name: str
    primitive: CPrimitive
    count: int = 1
    reserved: bool = False
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "primitive", CPrimitive(self.primitive))
        if self.schema_version != 1:
            raise SchemaError("unsupported C field schema version")
        if not self.name or self.count <= 0:
            raise SchemaError("field name and positive count required")


@dataclass(frozen=True)
class CStructABI(_Record):
    name: str
    fields: Tuple[CFieldABI, ...]
    alignment: int = 8
    endianness: str = "little"
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        if len({f.name for f in self.fields}) != len(self.fields):
            raise SchemaError("field names must be unique")
        if self.endianness != "little":
            raise SchemaError("C ABI must be little-endian")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise SchemaError("struct alignment must be power of two")
        if any(f.primitive.size_bytes > self.alignment for f in self.fields):
            raise SchemaError("struct alignment is smaller than natural field alignment")

    @property
    def field_offsets(self):
        offset, result = 0, []
        for f in self.fields:
            align = f.primitive.size_bytes
            offset = (offset + align - 1) // align * align
            result.append((f.name, offset))
            offset += f.count * align
        return tuple(result)

    @property
    def size_bytes(self):
        end = max((offset + f.count * f.primitive.size_bytes
                   for f, (_, offset) in zip(self.fields, self.field_offsets)), default=0)
        return (end + self.alignment - 1) // self.alignment * self.alignment

    def pack(self, values):
        unknown = set(values) - {f.name for f in self.fields}
        if unknown:
            raise SchemaError("unknown C ABI fields")
        payload = bytearray(self.size_bytes)
        for f, (_, offset) in zip(self.fields, self.field_offsets):
            if f.reserved:
                value = values.get(f.name, 0 if f.count == 1 else (0,) * f.count)
            elif f.name not in values:
                raise SchemaError(f"missing field {f.name}")
            else:
                value = values[f.name]
            items = (value,) if f.count == 1 else tuple(value)
            if len(items) != f.count:
                raise SchemaError(f"{f.name} requires {f.count} values")
            if f.reserved and any(items):
                raise SchemaError("reserved field must be zero")
            try:
                struct.pack_into("<" + f.primitive.struct_format * f.count,
                                 payload, offset, *items)
            except (struct.error, TypeError) as exc:
                raise SchemaError(str(exc)) from exc
        return bytes(payload)

    def unpack(self, payload):
        if len(payload) != self.size_bytes:
            raise SchemaError("C ABI payload size mismatch")
        values = {}
        occupied = set()
        for f, (_, offset) in zip(self.fields, self.field_offsets):
            items = struct.unpack_from("<" + f.primitive.struct_format * f.count,
                                       payload, offset)
            if f.reserved and any(items):
                raise SchemaError("reserved field must be zero")
            occupied.update(range(offset, offset + f.count * f.primitive.size_bytes))
            values[f.name] = items[0] if f.count == 1 else items
        if any(payload[i] for i in range(len(payload)) if i not in occupied):
            raise SchemaError("reserved padding must be zero")
        return values

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["fields"] = tuple(CFieldABI.from_dict(v) for v in values["fields"])
        return super().from_dict(values)


class KernelArgumentPassing(str, Enum):
    POINTER = "pointer"
    VALUE = "value"
    OPAQUE_HANDLE = "opaque_handle"


class KernelArgumentDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass(frozen=True)
class KernelArgumentABI(_Record):
    name: str
    passing: KernelArgumentPassing
    direction: KernelArgumentDirection
    primitive: CPrimitive = CPrimitive.U64
    nullable: bool = False
    required_alignment: int = 1
    pointee_abi_name: Optional[str] = None
    pointee_abi_fingerprint: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "passing", KernelArgumentPassing(self.passing))
        object.__setattr__(self, "direction", KernelArgumentDirection(self.direction))
        object.__setattr__(self, "primitive", CPrimitive(self.primitive))
        if self.nullable and self.passing != KernelArgumentPassing.POINTER:
            raise SchemaError("only pointer arguments may be nullable")
        if self.passing != KernelArgumentPassing.POINTER and self.direction != KernelArgumentDirection.INPUT:
            raise SchemaError("value and handle arguments must be input-only")
        if self.required_alignment <= 0 or self.required_alignment & (self.required_alignment - 1):
            raise SchemaError("argument alignment must be power of two")
        if (self.pointee_abi_name is None) != (self.pointee_abi_fingerprint is None):
            raise SchemaError("pointee ABI name/fingerprint must be paired")
        if self.pointee_abi_fingerprint is not None:
            _hash_field("pointee ABI", self.pointee_abi_fingerprint)


@dataclass(frozen=True)
class KernelErrorCodeABI(_Record):
    name: str
    code: int
    retryable: bool = False
    asynchronous: bool = False


@dataclass(frozen=True)
class KernelErrorABI(_Record):
    name: str
    codes: Tuple[KernelErrorCodeABI, ...]
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "codes", tuple(self.codes))
        if len({c.name for c in self.codes}) != len(self.codes) or len({c.code for c in self.codes}) != len(self.codes):
            raise SchemaError("error code names and values must be unique")
        if not any(c.name == "success" and c.code == 0 for c in self.codes):
            raise SchemaError("error ABI requires success=0")

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["codes"] = tuple(KernelErrorCodeABI.from_dict(v) for v in values["codes"])
        return super().from_dict(values)


@dataclass(frozen=True)
class KernelBinaryABI(_Record):
    abi_name: str
    arguments: Tuple[KernelArgumentABI, ...]
    error_abi: KernelErrorABI
    calling_convention: str = "c"
    return_primitive: CPrimitive = CPrimitive.I32
    pointer_width_bits: int = 64
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "return_primitive", CPrimitive(self.return_primitive))
        if self.return_primitive != CPrimitive.I32:
            raise SchemaError("kernel error return must be i32")
        if self.calling_convention != "c" or self.pointer_width_bits != 64:
            raise SchemaError("binary ABI requires C calling convention and 64-bit pointers")
        if len({a.name for a in self.arguments}) != len(self.arguments):
            raise SchemaError("binary argument names must be unique")

    def validate_logical(self, logical):
        if tuple(a.name for a in self.arguments) != logical.argument_names:
            raise SchemaError("binary/logical argument order mismatch")
        mutation = {a.name for a in self.arguments if a.direction != KernelArgumentDirection.INPUT}
        if mutation != set(logical.mutable_arguments):
            raise SchemaError("binary/logical mutation set mismatch")
        stream = next(a for a in self.arguments if a.name == logical.stream_argument)
        if stream.passing != KernelArgumentPassing.OPAQUE_HANDLE:
            raise SchemaError("stream must be opaque handle")
        if self.pointer_width_bits != logical.pointer_width_bits:
            raise SchemaError("binary/logical pointer width mismatch")

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["arguments"] = tuple(KernelArgumentABI.from_dict(v) for v in values["arguments"])
        values["error_abi"] = KernelErrorABI.from_dict(values["error_abi"])
        return super().from_dict(values)


@dataclass(frozen=True)
class ShapeRange(_Record):
    """某个张量轴上的非负整数闭区间；``name`` 用于可读的淘汰原因。

    ``tensor_index`` 按 ``build_signature`` 的张量参数顺序从 0 编号，
    ``axis`` 支持 Python 负索引。例如 ``ShapeRange("head_dim", 0, -1, 64, 128)``
    约束第一个张量的最后一维；page_size / seq_len 等由登记者按实际布局指定轴，
    不根据 op 或维度名称猜测布局。``minimum == maximum`` 表示仅支持一个值。
    """

    name: str
    tensor_index: int
    axis: int
    minimum: int
    maximum: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("shape range name must be non-empty")
        for name in ("tensor_index", "axis", "minimum", "maximum"):
            if type(getattr(self, name)) is not int:
                raise SchemaError(f"shape range {name} must be an integer")
        if self.tensor_index < 0:
            raise SchemaError("shape range tensor_index must be non-negative")
        if self.minimum < 0 or self.maximum < self.minimum:
            raise SchemaError("shape range requires 0 <= minimum <= maximum")


def _dtype_signatures(values, name="dtype_signatures"):
    if isinstance(values, (str, bytes, Mapping)):
        raise SchemaError(f"{name} must contain complete signatures")
    try:
        result = tuple(_feature_strings(name, v) for v in values)
    except TypeError as exc:
        raise SchemaError(f"{name} must contain complete signatures") from exc
    if any(not v for v in result) or len(set(result)) != len(result):
        raise SchemaError(f"{name} must contain unique non-empty signatures")
    return tuple(sorted(result))


def _shape_range_reasons(ranges, shapes):
    reasons = []
    for bound in ranges:
        interval = "[%d, %d]" % (bound.minimum, bound.maximum)
        if bound.tensor_index >= len(shapes):
            reasons.append("%s 缺少输入张量 %d 的 shape，无法检查支持区间 %s"
                           % (bound.name, bound.tensor_index, interval))
            continue
        shape = shapes[bound.tensor_index]
        if not -len(shape) <= bound.axis < len(shape):
            reasons.append("%s 的输入张量 %d 无轴 %d（shape=%s），无法检查支持区间 %s"
                           % (bound.name, bound.tensor_index, bound.axis, shape, interval))
            continue
        if not bound.minimum <= shape[bound.axis] <= bound.maximum:
            reasons.append("%s %d 不在支持区间 %s" % (bound.name, shape[bound.axis], interval))
    return tuple(reasons)


@dataclass(frozen=True)
class CapabilityCase(_Record):
    """One coherent AND clause. CapabilitySpec.cases is their OR.

    Requested semantic features require explicit declarations. Quantization,
    fusion and auxiliary outputs match exactly; they are not Cartesian sets.
    ``features`` compares catalogued functional choices without operator-specific
    dispatcher branches. Existing shape checks retain their contract role.
    """

    name: str
    dtype_signatures: Tuple[Tuple[str, ...], ...] = ()
    layout_signatures: Tuple[Tuple[str, ...], ...] = ()
    shape_ranges: Tuple[ShapeRange, ...] = ()
    shape_multiples: Tuple[Tuple[int, int, int], ...] = ()
    specializations: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    compute_dtypes: Tuple[str, ...] = ()
    accumulator_dtypes: Tuple[str, ...] = ()
    quantization: Tuple[Tuple[int, QuantSpec], ...] = ()
    attention_modes: Tuple[str, ...] = ()
    attention_variants: Tuple[str, ...] = ()
    kv_cache_types: Tuple[str, ...] = ()
    mask_types: Tuple[str, ...] = ()
    pos_encodings: Tuple[str, ...] = ()
    causal_values: Tuple[bool, ...] = ()
    sliding_window_values: Tuple[bool, ...] = ()
    supports_logits_soft_cap: bool = False
    fusion_ops: Tuple[str, ...] = ()
    auxiliary_outputs: Tuple[str, ...] = ()
    sparse_formats: Tuple[str, ...] = ()
    routing_methods: Tuple[str, ...] = ()
    moe_top_k_values: Tuple[int, ...] = ()
    expert_parallel_sizes: Tuple[int, ...] = ()
    supports_deterministic: bool = False
    features: Tuple[FeatureConstraint, ...] = ()
    quantization_group_size_ranges: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaError("capability case name must be non-empty")
        for name in ("dtype_signatures", "layout_signatures"):
            object.__setattr__(self, name, _dtype_signatures(getattr(self, name), name))
        for name in ("compute_dtypes", "accumulator_dtypes", "attention_modes", "attention_variants",
                     "kv_cache_types", "mask_types", "pos_encodings", "sparse_formats", "routing_methods"):
            object.__setattr__(self, name, tuple(sorted(set(_feature_strings(name, getattr(self, name))))))
        for name in ("fusion_ops", "auxiliary_outputs"):
            object.__setattr__(self, name, _feature_strings(name, getattr(self, name)))
        for name in ("causal_values", "sliding_window_values", "moe_top_k_values", "expert_parallel_sizes"):
            try:
                values = tuple(getattr(self, name))
            except TypeError as exc:
                raise SchemaError(f"invalid {name}") from exc
            boolean = name in ("causal_values", "sliding_window_values")
            if any(type(v) is not (bool if boolean else int) or (not boolean and v <= 0) for v in values):
                raise SchemaError(f"invalid {name}")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        for name in ("supports_logits_soft_cap", "supports_deterministic"):
            if type(getattr(self, name)) is not bool:
                raise SchemaError(f"{name} must be bool")
        # Reuse the existing axis validation and canonical ordering.
        try:
            ranges = tuple(self.shape_ranges)
        except TypeError as exc:
            raise SchemaError("case shape_ranges must contain ShapeRange values") from exc
        if any(not isinstance(r, ShapeRange) for r in ranges):
            raise SchemaError("case shape_ranges must contain ShapeRange values")
        if len({(r.tensor_index, r.axis) for r in ranges}) != len(ranges):
            raise SchemaError("case shape_ranges must not repeat a tensor axis")
        object.__setattr__(self, "shape_ranges", tuple(sorted(ranges, key=lambda r: (r.tensor_index, r.axis))))
        try:
            multiples = tuple(tuple(v) for v in self.shape_multiples)
            valid = all(len(v) == 3 and all(type(n) is int for n in v)
                        and v[0] >= 0 and v[2] > 0 for v in multiples)
        except TypeError as exc:
            raise SchemaError("invalid shape_multiples") from exc
        if not valid or len({v[:2] for v in multiples}) != len(multiples):
            raise SchemaError("shape_multiples require unique (tensor_index, axis, positive multiple)")
        object.__setattr__(self, "shape_multiples", tuple(sorted(multiples)))
        object.__setattr__(self, "quantization", _quantization_bindings(self.quantization))
        ranges = tuple(tuple(v) for v in self.quantization_group_size_ranges)
        quantization = dict(self.quantization)
        if (any(len(v) != 3 or any(type(n) is not int for n in v)
                or v[0] < 0 or v[1] < 1 or v[2] < v[1]
                or v[0] not in quantization or quantization[v[0]].granularity != "group"
                or quantization[v[0]].group_size != (v[1],) for v in ranges)
                or len({v[0] for v in ranges}) != len(ranges)):
            raise SchemaError("quantization group ranges require unique operand, positive bounds and matching group lower bound")
        object.__setattr__(self, "quantization_group_size_ranges", tuple(sorted(ranges)))
        try:
            specializations = tuple((k, tuple(sorted(set(_feature_strings("allowed values", v)))))
                                    for k, v in self.specializations)
        except (TypeError, ValueError) as exc:
            raise SchemaError("invalid case specializations") from exc
        if (any(not isinstance(k, str) or not k.strip() or not v for k, v in specializations)
                or len({k for k, _ in specializations}) != len(specializations)):
            raise SchemaError("case specializations require unique names and allowed values")
        object.__setattr__(self, "specializations", tuple(sorted(specializations)))
        try:
            features = tuple(self.features)
        except TypeError as error:
            raise SchemaError("features must contain FeatureConstraint values") from error
        if any(not isinstance(feature, FeatureConstraint) for feature in features):
            raise SchemaError("features must contain FeatureConstraint values")
        if len({feature.name for feature in features}) != len(features):
            raise SchemaError("features must not repeat a name")
        object.__setattr__(self, "features", tuple(sorted(features, key=lambda item: item.name)))
        self.feature_constraints()

    def feature_constraints(self):
        """Merge legacy aliases without expanding the generic matching algorithm."""
        allowed = {feature.name: feature.allowed_values for feature in self.features}
        for _, key, field in INFERENCE_FEATURE_ALIASES:
            values = getattr(self, field)
            if values:
                if key in allowed and set(allowed[key]) != set(values):
                    raise SchemaError(f"feature {key} contradicts {field}")
                allowed[key] = values
        return allowed

    def to_dict(self):
        value = super().to_dict()
        if not self.features:
            value.pop("features")
        if not self.quantization_group_size_ranges:
            value.pop("quantization_group_size_ranges")
        return value

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, Mapping) or "name" not in value
                or not set(value) <= {f.name for f in fields(cls)}):
            raise SchemaError("CapabilityCase fields do not match schema")
        try:
            values = dict(value)
            values["quantization"] = _decode_quantization(values.get("quantization", ()))
            values["shape_ranges"] = tuple(ShapeRange.from_dict(v) for v in values.get("shape_ranges", ()))
            values["features"] = tuple(FeatureConstraint.from_dict(v) for v in values.get("features", ()))
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc

    def unsupported_reasons(self, signature):
        actual = signature.inference or InferenceSpec()
        reasons = list(_shape_range_reasons(self.shape_ranges, signature.shapes))
        for name in ("dtype", "layout"):
            allowed = getattr(self, name + "_signatures")
            if allowed and getattr(signature, name + "s") not in allowed:
                reasons.append(f"unsupported {name}_signature {getattr(signature, name + 's')!r}")
        for index, axis, multiple in self.shape_multiples:
            if index >= len(signature.shapes) or not -len(signature.shapes[index]) <= axis < len(signature.shapes[index]):
                reasons.append(f"shape_multiples missing tensor {index} axis {axis}")
            elif signature.shapes[index][axis] % multiple:
                reasons.append(f"shape tensor {index} axis {axis} must be a multiple of {multiple}")
        facts, constraints = dict(actual.feature_values()), self.feature_constraints()
        aliases = {key: field for field, key, _ in INFERENCE_FEATURE_ALIASES}
        for name in sorted(facts.keys() | constraints.keys()):
            value, allowed = facts.get(name), constraints.get(name, ())
            if name not in facts or value not in allowed:
                label = f"{name} ({aliases[name]})" if name in aliases else name
                reasons.append(f"{label}={value!r} unsupported or missing; allowed={allowed!r}")
        expected_quantization = dict(self.quantization)
        actual_quantization = dict(actual.quantization)
        for index, minimum, maximum in self.quantization_group_size_ranges:
            spec = actual_quantization.get(index)
            size = None if spec is None else spec.group_size
            if size is None or len(size) != 1 or not minimum <= size[0] <= maximum:
                reasons.append(f"quantization operand {index} group_size outside [{minimum}, {maximum}]")
            else:
                expected_quantization[index] = replace(expected_quantization[index], group_size=size)
        if actual.quantization != tuple(sorted(expected_quantization.items())):
            reasons.append("unsupported exact quantization")
        for name in ("fusion_ops", "auxiliary_outputs"):
            if getattr(actual, name) != getattr(self, name):
                reasons.append(f"unsupported exact {name}")
        if any(index >= len(signature.dtypes) for index, _ in actual.quantization):
            reasons.append("quantization tensor_index has no dtype metadata")
        if actual.logits_soft_cap and not self.supports_logits_soft_cap:
            reasons.append("logits_soft_cap is unsupported")
        if (actual.deterministic or ("deterministic", "True") in signature.specializations) and not self.supports_deterministic:
            reasons.append("deterministic execution is unsupported")
        for key, allowed in self.specializations:
            values = tuple(v for k, v in signature.specializations if k == key)
            if not values or any(v not in allowed for v in values):
                reasons.append(f"specialization {key} missing or unsupported; allowed={allowed!r}")
        return tuple(reasons)


@dataclass(frozen=True)
class CapabilitySpec(_Record):
    """kernel 的能力声明：能对哪些 (输入签名 × 硬件签名) 执行。

    空集合表示"不限制"。``specializations`` 逐项给出允许值集合——
    签名中出现的键若声明了允许值则必须命中，未声明的键不受限。
    ``shape_ranges`` 独立声明张量各轴的闭区间，所有区间必须同时满足；
    声明的张量或轴缺失时淘汰，不把未知规模当作支持。
    ``physical_layouts`` 是所有张量输入允许的物理排布集合，仅支持 ND / NZ
    （NZ 精确对应 FRACTAL_NZ，不合并其他分形格式）；非空时未知排布也淘汰。
    它与 NHD / HND 等逻辑 ``layouts`` 独立，不执行排布转换。
    ``dtype_signatures`` 匹配按参数顺序排列的完整 dtype 组合。
    ``cases`` 表达多个连贯的能力组合：同一 case 内 AND，case 之间 OR，
    并与外层限制 AND。显式 InferenceSpec 必须有 case 声明才可通过；
    不为历史 descriptor 猜测量化、融合、Attention 等语义能力。
    """

    op: str
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    soc_models: Tuple[str, ...] = ()
    min_aicore: int = 0
    specializations: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    shape_ranges: Tuple[ShapeRange, ...] = ()
    physical_layouts: Tuple[str, ...] = ()
    dtype_signatures: Tuple[Tuple[str, ...], ...] = ()
    cases: Tuple[CapabilityCase, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "dtype_signatures", _dtype_signatures(self.dtype_signatures))
        try:
            cases = tuple(self.cases)
        except TypeError as exc:
            raise SchemaError("cases must contain CapabilityCase values") from exc
        if any(not isinstance(case, CapabilityCase) for case in cases):
            raise SchemaError("cases must contain CapabilityCase values")
        if len({case.name for case in cases}) != len(cases):
            raise SchemaError("capability case names must be unique")
        object.__setattr__(self, "cases", tuple(sorted(cases, key=lambda case: case.name)))
        try:
            layouts = tuple(self.physical_layouts)
        except TypeError as exc:
            raise SchemaError("physical_layouts must contain ND / NZ values") from exc
        if isinstance(self.physical_layouts, str) or any(
            not isinstance(layout, str) or layout not in ("ND", "NZ")
            for layout in layouts
        ):
            raise SchemaError("physical_layouts must contain ND / NZ values")
        object.__setattr__(self, "physical_layouts", tuple(sorted(set(layouts))))
        try:
            ranges = tuple(self.shape_ranges)
        except TypeError as exc:
            raise SchemaError("shape_ranges must contain ShapeRange values") from exc
        if any(not isinstance(r, ShapeRange) for r in ranges):
            raise SchemaError("shape_ranges must contain ShapeRange values")
        axes = [(r.tensor_index, r.axis) for r in ranges]
        if len(set(axes)) != len(axes):
            raise SchemaError("shape_ranges must not repeat a tensor axis")
        object.__setattr__(self, "shape_ranges", tuple(sorted(
            ranges, key=lambda r: (r.tensor_index, r.axis)
        )))

    def to_dict(self):
        value = super().to_dict()
        # 默认不限制，保留旧能力声明的序列化形式与指纹。
        if not self.shape_ranges:
            value.pop("shape_ranges")
        if not self.physical_layouts:
            value.pop("physical_layouts")
        if not self.dtype_signatures:
            value.pop("dtype_signatures")
        if not self.cases:
            value.pop("cases")
        else:
            value["cases"] = [case.to_dict() for case in self.cases]
        return value

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, Mapping) or "op" not in value
                or not set(value) <= {f.name for f in fields(cls)}):
            raise SchemaError("CapabilitySpec fields do not match schema")
        values = dict(value)
        try:
            for name in ("dtypes", "layouts", "soc_models"):
                values[name] = tuple(values.get(name, ()))
            values["specializations"] = tuple(
                (key, tuple(allowed)) for key, allowed
                in values.get("specializations", ())
            )
            values["shape_ranges"] = tuple(
                ShapeRange.from_dict(r) for r in values.get("shape_ranges", ())
            )
            values["cases"] = tuple(CapabilityCase.from_dict(case) for case in values.get("cases", ()))
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc

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
        if self.dtype_signatures and signature.dtypes not in self.dtype_signatures:
            reasons.append("unsupported dtype_signature %r" % (signature.dtypes,))
        if self.cases:
            reports = [(case.name, case.unsupported_reasons(signature)) for case in self.cases]
            if all(failures for _, failures in reports):
                reasons.extend("capability case %s: %s" % (name, "; ".join(failures))
                               for name, failures in reports)
        elif signature.inference is not None:
            reasons.append("inference features require explicitly declared capability cases")
        if self.layouts:
            for layout in signature.layouts:
                if layout and layout not in self.layouts:
                    reasons.append(
                        "layout %s 不在支持集 %s" % (layout, sorted(self.layouts))
                    )
        if self.physical_layouts:
            allowed = sorted(self.physical_layouts)
            tensor_count = max(
                len(signature.dtypes), len(signature.shapes), len(signature.layouts),
                len(signature.physical_layouts),
            )
            if not tensor_count:
                reasons.append("缺少输入张量的物理排布，无法检查支持集 %s" % allowed)
            for index in range(tensor_count):
                layout = (
                    signature.physical_layouts[index]
                    if index < len(signature.physical_layouts) else ""
                )
                if not layout:
                    reasons.append(
                        "输入张量 %d 物理排布未知，无法检查支持集 %s" % (index, allowed)
                    )
                elif layout not in self.physical_layouts:
                    reasons.append(
                        "输入张量 %d 物理排布 %s 不在支持集 %s" % (index, layout, allowed)
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
        reasons.extend(_shape_range_reasons(self.shape_ranges, signature.shapes))
        return tuple(reasons)


# kernel 的可执行入口：reference 为 Python 可调用；自研 kernel 由
# AOT 产物加载 / JIT 编译流程填充（M0 阶段为 None，拉起时报错）。
KernelRunner = Callable[..., Any]


@dataclass(frozen=True)
class GraphReplaySpec(_Record):
    """Serializable NPU replay recipe and the finite scope actually validated.

    ``host_kwargs`` replaces named host arguments and reissues the saved
    operation inside a task update; it never replaces captured tensor storage.
    Evidence is bound to a kernel ID, not transferable to another descriptor.
    Public access is through :mod:`flashinfer.npu`.
    """

    validated_kernel_id: str
    validation_tests: Tuple[str, ...]
    validated_environment: str
    strategy: str = "host_kwargs"
    parameter_sources: Tuple[Tuple[str, str], ...] = ()
    required_arguments: Tuple[str, ...] = ()
    fixed_arguments: Tuple[Tuple[str, Any], ...] = ()
    tensor_metadata: Tuple[Tuple[str, Tuple[int, ...], Tuple[str, ...]], ...] = ()
    parameter_ranges: Tuple[Tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if not isinstance(self.strategy, str) or self.strategy not in {"host_kwargs", "none"}:
            raise SchemaError("unsupported graph replay strategy")
        if (not isinstance(self.validated_kernel_id, str) or not self.validated_kernel_id
                or not isinstance(self.validated_environment, str) or not self.validated_environment):
            raise SchemaError("graph replay requires kernel-bound validation evidence and environment")
        for name in ("validation_tests", "required_arguments"):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list))
                    or any(not isinstance(v, str) or not v.strip() for v in values)
                    or len(set(values)) != len(values)):
                raise SchemaError("invalid graph replay " + name)
            object.__setattr__(self, name, tuple(values))
        if not self.validation_tests:
            raise SchemaError("graph replay requires non-empty validation_tests")
        try:
            sources = tuple((name, source) for name, source in self.parameter_sources)
            fixed = tuple((name, value) for name, value in self.fixed_arguments)
            metadata = tuple((name, tuple(shape), tuple(dtypes))
                             for name, shape, dtypes in self.tensor_metadata)
            ranges = tuple((name, lo, hi) for name, lo, hi in self.parameter_ranges)
        except (TypeError, ValueError) as exc:
            raise SchemaError("invalid graph replay argument constraints") from exc
        for values in (sources, fixed, metadata, ranges):
            if (any(not isinstance(v[0], str) or not v[0].strip() for v in values)
                    or len({v[0] for v in values}) != len(values)):
                raise SchemaError("graph replay argument constraints require unique names")
        if any(not isinstance(source, str) or not source.strip() for _, source in sources):
            raise SchemaError("graph replay parameter sources require non-empty names")
        if any(type(v) not in (bool, int, float, str, type(None)) for _, v in fixed):
            raise SchemaError("graph replay fixed arguments must be host scalars")
        if any(not shape or any(type(d) is not int or d <= 0 for d in shape)
               or not dtypes or any(not isinstance(d, str) or not d for d in dtypes)
               for _, shape, dtypes in metadata):
            raise SchemaError("invalid graph replay tensor metadata")
        if any(type(lo) is not int or type(hi) is not int or lo > hi for _, lo, hi in ranges):
            raise SchemaError("invalid graph replay parameter ranges")
        object.__setattr__(self, "parameter_sources", sources)
        object.__setattr__(self, "fixed_arguments", fixed)
        object.__setattr__(self, "tensor_metadata", metadata)
        object.__setattr__(self, "parameter_ranges", ranges)


@dataclass(frozen=True)
class KernelDescriptor:
    """registry 条目：一个可执行 kernel 的全部静态描述（§11）。

    ``priority`` 用于确定性启发式的同层 tie-break（维护者声明，
    新版本号优先，§5.4.3）。

    ``artifact`` 指向已编译产物，``source`` 指向可编译源码，允许两者并存。
    ``target_soc`` 是尚无产物时的编译目标；None 表示未指定，已有产物时
    以 ``artifact.target_soc`` 为准。不同 SOC 的产物分别使用唯一 kernel_id
    登记（例如 ``ascendc/op/Ascend910B``），不改变 registry 的身份规则。

    ``graph_capability`` 默认 ``NOT_CAPTURABLE``：未经入图验证的记录不能
    据此进入图，但仍正常参与 eager 选型。``NEEDS_PARAM_UPDATE`` 必须声明
    非空的 ``graph_param_updates``，列出重放前须改写的 kernel 入参符号名；
    元组顺序不表示 ABI 参数位置。``graph_replay`` 声明统一更新策略、已验证
    范围及测试证据，由 T11 消费；没有证据的静态声明不能用于 NPU 图接口。
    """

    kernel_id: str
    op: str = ""
    backend: Backend = Backend.REFERENCE
    capability: Optional[CapabilitySpec] = None
    compile_mode: CompileMode = CompileMode.NONE
    priority: int = 0
    runner: Optional[KernelRunner] = None
    constraints: KernelConstraints = KernelConstraints()
    workspace: WorkspaceFormula = WorkspaceFormula()
    int_workspace: WorkspaceFormula = WorkspaceFormula()
    artifact: Optional[ArtifactRef] = None
    launch_abi: Optional[KernelLaunchABI] = None
    binary_abi: Optional[KernelBinaryABI] = None
    op_schema_version: int = 1
    tiling_schema_version: int = 1
    capability_binding: Optional[KernelCapabilityBinding] = None
    source: Optional[SourceRef] = None
    target_soc: Optional[str] = None
    graph_capability: GraphCapability = GraphCapability.NOT_CAPTURABLE
    graph_param_updates: Tuple[str, ...] = ()
    graph_replay: Optional[GraphReplaySpec] = None
    admission: Optional[Callable] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "backend", Backend(self.backend))
        if self.admission is not None and not callable(self.admission):
            raise SchemaError("kernel admission must be callable")
        try:
            object.__setattr__(self, "compile_mode", CompileMode(self.compile_mode))
        except (TypeError, ValueError) as exc:
            raise SchemaError("invalid compile_mode") from exc
        try:
            graph_capability = GraphCapability(self.graph_capability)
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                "graph_capability must be capture_safe, needs_param_update, or not_capturable"
            ) from exc
        object.__setattr__(self, "graph_capability", graph_capability)
        if not isinstance(self.graph_param_updates, (tuple, list)):
            raise SchemaError("graph_param_updates must be a list or tuple of parameter names")
        params = tuple(self.graph_param_updates)
        if any(
            not isinstance(name, str) or not name
            or any(c.isspace() or c == "\0" for c in name)
            for name in params
        ):
            raise SchemaError(
                "graph_param_updates must contain non-empty parameter names without whitespace or NUL"
            )
        if len(set(params)) != len(params):
            raise SchemaError("graph_param_updates must not repeat a parameter name")
        if graph_capability == GraphCapability.NEEDS_PARAM_UPDATE:
            if not params:
                raise SchemaError("needs_param_update requires non-empty graph_param_updates")
        elif params:
            raise SchemaError("graph_param_updates requires graph_capability=needs_param_update")
        object.__setattr__(self, "graph_param_updates", params)
        if self.graph_replay is not None:
            spec = self.graph_replay
            if not isinstance(spec, GraphReplaySpec):
                raise SchemaError("graph_replay must be GraphReplaySpec or None")
            if spec.validated_kernel_id != self.kernel_id:
                raise SchemaError("graph replay evidence belongs to a different kernel")
            if graph_capability == GraphCapability.NOT_CAPTURABLE:
                raise SchemaError("not_capturable kernel cannot declare graph_replay")
            expected = "host_kwargs" if graph_capability == GraphCapability.NEEDS_PARAM_UPDATE else "none"
            if spec.strategy != expected:
                raise SchemaError("graph replay strategy does not match graph_capability")
            if {name for name, _ in spec.parameter_sources} != set(params):
                raise SchemaError("graph replay parameter_sources must bind exactly graph_param_updates")
            if any(name not in params for name, _, _ in spec.parameter_ranges):
                raise SchemaError("graph replay ranges must name graph_param_updates")
        if self.capability is None:
            object.__setattr__(self, "capability", CapabilitySpec(op=self.op))
        if not self.op:
            object.__setattr__(self, "op", self.capability.op)
        if not self.kernel_id or not self.op:
            raise SchemaError("kernel_id and op must be non-empty")
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise SchemaError("source must be SourceRef or None")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactRef):
            raise SchemaError("artifact must be ArtifactRef or None")
        if self.target_soc is not None and (
            not isinstance(self.target_soc, str) or not self.target_soc
            or any(c.isspace() or c == "\0" for c in self.target_soc)
        ):
            raise SchemaError("target_soc must be a non-empty SOC identifier or None")
        if (self.artifact is not None and self.target_soc is not None
                and self.target_soc != self.artifact.target_soc):
            raise SchemaError("target_soc does not match artifact target_soc")
        formats = {
            Backend.ASCENDC_AOT: (ArtifactFormat.ASCENDC_OBJECT, ArtifactFormat.SHARED_LIBRARY,
                                 ArtifactFormat.ASCENDC_OP_PACKAGE),
            Backend.ASCENDC_JIT: (ArtifactFormat.ASCENDC_SOURCE,),
            Backend.ACLNN: (ArtifactFormat.ACLNN_BUILTIN,),
        }
        if self.backend in formats:
            if not isinstance(self.artifact, ArtifactRef):
                raise SchemaError("non-reference kernel requires ArtifactRef")
            if not isinstance(self.launch_abi, KernelLaunchABI):
                raise SchemaError("non-reference kernel requires KernelLaunchABI")
            if not isinstance(self.binary_abi, KernelBinaryABI):
                raise SchemaError("non-reference kernel requires KernelBinaryABI")
            if self.artifact.format not in formats[self.backend]:
                raise SchemaError("artifact format does not match backend")
            self.binary_abi.validate_logical(self.launch_abi)
        elif self.backend == Backend.REFERENCE and any(
            v is not None for v in (self.artifact, self.launch_abi, self.binary_abi)
        ):
            raise SchemaError("reference kernel cannot declare artifact or launch ABI")
        if self.artifact is not None:
            if self.launch_abi is not None and not isinstance(self.launch_abi, KernelLaunchABI):
                raise SchemaError("launch_abi must be KernelLaunchABI or None")
            if self.artifact.entry_point is None and self.launch_abi is None:
                raise SchemaError("artifact requires entry_point or KernelLaunchABI")
            if (self.artifact.entry_point is not None and self.launch_abi is not None
                    and self.artifact.entry_point != self.launch_abi.entry_point):
                raise SchemaError("artifact entry_point does not match launch ABI")

    def admission_reasons(self, signature) -> Tuple[str, ...]:
        """Apply the bound implementation's metadata contract before ranking.

        Like runner, this installed Python binding is excluded from serialized
        ABI fields. Its native recipe remains bound by the frozen build spec.
        """
        reasons = () if self.admission is None else tuple(self.admission(signature))
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise SchemaError("kernel admission must return nonempty reason strings")
        return reasons

    def unsupported_target_reasons(self, soc_model: str) -> Tuple[str, ...]:
        """编译目标必须匹配设备；capability 的宽支持集不能放宽产物目标。"""
        target = self.artifact.target_soc if self.artifact is not None else self.target_soc
        if target is not None and target != soc_model:
            return ("target SoC %s does not match device SoC %s" % (target, soc_model),)
        return ()

    def to_dict(self):
        value = {
            name: _json_value(getattr(self, name)) for name in (
                "kernel_id", "op", "backend", "constraints", "workspace", "int_workspace",
                "artifact", "launch_abi", "binary_abi", "priority", "op_schema_version",
                "tiling_schema_version", "capability_binding",
            )
        }
        if self.artifact is not None:
            value["artifact"] = self.artifact.to_dict()
        # 新的默认字段不写入旧记录，避免无关的指纹迁移。
        if self.source is not None:
            value["source"] = self.source.to_dict()
        if self.target_soc is not None:
            value["target_soc"] = self.target_soc
        if self.compile_mode != CompileMode.NONE:
            value["compile_mode"] = self.compile_mode.value
        if self.graph_capability != GraphCapability.NOT_CAPTURABLE:
            value["graph_capability"] = self.graph_capability.value
        if self.graph_param_updates:
            value["graph_param_updates"] = list(self.graph_param_updates)
        if self.graph_replay is not None:
            value["graph_replay"] = self.graph_replay.to_dict()
        # 历史记录未序列化 capability；规模或物理排布受限的新记录携带它，
        # 保证限制进入静态身份且可往返，同时不迁移旧记录的指纹。
        if (self.capability.shape_ranges or self.capability.physical_layouts
                or self.capability.dtype_signatures or self.capability.cases):
            value["capability"] = self.capability.to_dict()
        return value

    @property
    def fingerprint(self):
        return _fingerprint(self.to_dict())
