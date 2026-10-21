"""runtime 静态契约：Backend / CapabilitySpec / KernelDescriptor（§5、§11）。

设计要点（§5.1）：调度涉及 op / backend / kernel 三个粒度递进的概念——
- op      语义层（"做什么"），如 ``attention.decode``
- backend 实现来源层，是 kernel 的分类属性，不是独立决策对象
- kernel  唯一可执行单元，调度决策只在 kernel 粒度上发生
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
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
    # Historical artifact/provider contracts retain their serialized identities.
    ASCENDC_AOT = "ascendc_aot"
    ASCENDC_JIT = "ascendc_jit"
    ACLNN = "aclnn"


class CompileMode(str, Enum):
    """自研 kernel 的编译方式（§7）；CANN / reference 取 ``NONE``。"""

    AOT = "aot"
    JIT = "jit"
    NONE = "none"


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

    def __post_init__(self):
        if not self.op:
            raise SchemaError("workload op must be non-empty")
        for name in ("dtypes", "layouts", "static_dims", "dynamic_bounds", "quant_specs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "attributes", tuple(sorted(
            (str(k), str(v)) for k, v in self.attributes
        )))

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        values["quant_specs"] = tuple(QuantSpec.from_dict(v) for v in values["quant_specs"])
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
    SHARED_LIBRARY = "shared_library"
    ASCENDC_SOURCE = "ascendc_source"
    ACLNN_BUILTIN = "aclnn_builtin"


@dataclass(frozen=True)
class ArtifactRef(_Record):
    kind: ArtifactKind
    format: ArtifactFormat
    locator: str
    digest: str
    target_soc: str
    build_id: str
    size_bytes: Optional[int] = None
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "kind", ArtifactKind(self.kind))
        object.__setattr__(self, "format", ArtifactFormat(self.format))
        if self.schema_version != 1:
            raise SchemaError("unsupported artifact schema_version")
        _hash_field("artifact digest", self.digest)
        if not self.target_soc or not self.build_id or not self.locator:
            raise SchemaError("artifact target_soc, build_id and locator required")
        expected = {
            ArtifactFormat.ASCENDC_OBJECT: ArtifactKind.FILE,
            ArtifactFormat.SHARED_LIBRARY: ArtifactKind.FILE,
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

    def __post_init__(self):
        object.__setattr__(self, "backend", Backend(self.backend))
        if self.capability is None:
            object.__setattr__(self, "capability", CapabilitySpec(op=self.op))
        if not self.op:
            object.__setattr__(self, "op", self.capability.op)
        if not self.kernel_id or not self.op:
            raise SchemaError("kernel_id and op must be non-empty")
        formats = {
            Backend.ASCENDC_AOT: (ArtifactFormat.ASCENDC_OBJECT, ArtifactFormat.SHARED_LIBRARY),
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

    def to_dict(self):
        return {
            name: _json_value(getattr(self, name)) for name in (
                "kernel_id", "op", "backend", "constraints", "workspace", "int_workspace",
                "artifact", "launch_abi", "binary_abi", "priority", "op_schema_version",
                "tiling_schema_version", "capability_binding",
            )
        }

    @property
    def fingerprint(self):
        return _fingerprint(self.to_dict())
