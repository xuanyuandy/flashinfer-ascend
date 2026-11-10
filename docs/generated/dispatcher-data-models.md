# Dispatcher 数据结构定义与关联索引

本页从当前源码自动提取，字段逐项与实际 dataclass 核对。代码块列出完整字段（继承字段展开），省略方法、校验和 docstring；它们不是 C++ 内存布局。

先读 PDF 的组件章节，再用本页查字段。`InputSignature` 是请求事实；`KernelRegistry` 是保存 `KernelDescriptor` 的容器；`dispatch()` 是函数，输出 `DispatchDecision`。

[Matmul 接口与调用链](../matmul-backends.md) · [架构讲解 PDF](../presentations/dispatcher.pdf)

实际校验：35 个 dataclass、254 个字段、8 个枚举；5 次真实 dispatch，包含决策缓存命中。样例未申请 Tensor、未执行 kernel，生产 registry 保持不变。

## 容器与控制值

`KernelRegistry` 是可变 Python 类，内部保存如下成员；内置登记冻结后通过只读查询使用，合法的 external/custom 登记更新 revision。

```python
class KernelRegistry:
    def __init__(self, descriptors=()):
        self._descriptors: Dict[str, KernelDescriptor] = {}
        self._frozen = False
        self._revision = 0
        self._lock = threading.Lock()
```

常用操作：`register`/`freeze` 完成初始化；`descriptors(op)` 按逻辑算子取候选；`stacks(op)` 派生来源视图；`get(kernel_id)` 连接最终决策与 runner。

`policy` 是 `auto` 或 `Backend` 值，`order` 是档位元组，不存在专门的 BackendPolicy 类。调优 record 是字典，查询的公共返回值为 `Optional[str]` 冠军 ID。

决策缓存外层是 `WeakKeyDictionary[KernelRegistry, Dict[tuple, DispatchDecision]]`；内层 key 的实际字段如下：

```python
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
    tuning_cache_token(),
)
```

## 一条 Matmul 请求的实际数据

私有 registry 复制生产 Matmul 的三个 descriptor，并去掉 runner 以隔离执行；能力、身份与排序元数据保持相同。

```json
{
  "signature": {
    "op": "gemm.matmul",
    "dtypes": [
      "bfloat16",
      "bfloat16",
      "bfloat16"
    ],
    "shapes": [
      [
        32,
        48
      ],
      [
        48,
        64
      ],
      [
        32,
        64
      ]
    ],
    "layouts": [
      "",
      "",
      ""
    ],
    "specializations": [
      [
        "tensor_contract",
        "supported"
      ]
    ],
    "physical_layouts": [
      "ND",
      "ND",
      "ND"
    ]
  },
  "hardware": {
    "soc_model": "Ascend950",
    "aicore_count": 56,
    "ub_size_bytes": 221184
  }
}
```

默认 auto 的实际结果为 CANN；三候选均 accepted，只有 cann 栈 selected。相同请求第二次 cache_hit=True；强制 cann/ascendc/triton 分别选择对应来源。

## 完整 dataclass 字段

### InputSignature

facade 创建本次调用的元数据快照；能力过滤、调优查询和决策缓存消费它。不是 Tensor 或产物。

源码：[flashinfer/runtime/signature.py:14](../../flashinfer/runtime/signature.py)。

被引用位置：`Plan.signature`

```python
@dataclass(frozen=True)
class InputSignature:
    op: str
    dtypes: Tuple[str, ...] = ()
    shapes: Tuple[Tuple[int, ...], ...] = ()
    layouts: Tuple[str, ...] = ()
    specializations: Tuple[Tuple[str, str], ...] = ()
    physical_layouts: Tuple[str, ...] = ()
```

### HardwareSignature

hardware probe 创建 SoC / AI Core / UB 快照；过滤和调优匹配消费它，不包含 Tensor 地址。

源码：[flashinfer/runtime/hardware.py:20](../../flashinfer/runtime/hardware.py)。

被引用位置：`Plan.hardware`

```python
@dataclass(frozen=True)
class HardwareSignature:
    soc_model: str = 'unknown'
    aicore_count: int = 0
    ub_size_bytes: int = 0
```

### CapabilitySpec

算子作者声明一个实现的支持域；unsupported_reasons(signature, hardware) 对照实际请求。

源码：[flashinfer/runtime/schema.py:689](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.capability`

```python
@dataclass(frozen=True)
class CapabilitySpec:
    op: str
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    soc_models: Tuple[str, ...] = ()
    min_aicore: int = 0
    specializations: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    shape_ranges: Tuple[ShapeRange, ...] = ()
    physical_layouts: Tuple[str, ...] = ()
```

### ShapeRange

CapabilitySpec 的一个 Tensor 轴闭区间；tensor_index/axis 负责轴映射，name 只用于诊断。

源码：[flashinfer/runtime/schema.py:661](../../flashinfer/runtime/schema.py)。

被引用位置：`CapabilitySpec.shape_ranges`

```python
@dataclass(frozen=True)
class ShapeRange:
    name: str
    tensor_index: int
    axis: int
    minimum: int
    maximum: int
```

### KernelDescriptor

provider/manifest 创建一条实现记录；registry 保存它，dispatcher 选择它的身份，执行层调用其 runner。

源码：[flashinfer/runtime/schema.py:914](../../flashinfer/runtime/schema.py)。

被引用位置：`ImplementationStack.descriptors`

```python
@dataclass(frozen=True)
class KernelDescriptor:
    kernel_id: str
    op: str = ''
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
```

### ImplementationStack

registry.stacks 从 descriptor 派生的来源视图；两级选型和解释使用它，不另设栈能力或优先级。

源码：[flashinfer/runtime/registry.py:39](../../flashinfer/runtime/registry.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class ImplementationStack:
    stack_id: str
    descriptors: Tuple[KernelDescriptor, ...]
```

### CandidateReport

过滤阶段创建一个候选的 accepted/reasons；DispatchDecision.reports 保留逐候选结论。

源码：[flashinfer/runtime/dispatch.py:33](../../flashinfer/runtime/dispatch.py)。

被引用位置：`StackReport.implementations`、`DispatchDecision.reports`

```python
@dataclass(frozen=True)
class CandidateReport:
    kernel_id: str
    backend: Backend
    accepted: bool
    reasons: Tuple[str, ...] = ()
```

### StackReport

解释阶段创建来源栈资格和选中状态；implementations 关联候选报告。accepted 不代表 selected。

源码：[flashinfer/runtime/dispatch.py:43](../../flashinfer/runtime/dispatch.py)。

被引用位置：`DispatchDecision.stack_reports`

```python
@dataclass(frozen=True)
class StackReport:
    stack_id: str
    accepted: bool
    selected: bool = False
    representative_kernel_id: Optional[str] = None
    reasons: Tuple[str, ...] = ()
    implementations: Tuple[CandidateReport, ...] = ()
```

### DispatchDecision

dispatch 的最终输出；facade 读取 kernel_id，随后在 registry 中找到原 runner。它不保存输入 Tensor 或二进制。

源码：[flashinfer/runtime/dispatch.py:61](../../flashinfer/runtime/dispatch.py)。

被引用位置：`Plan.decision`

```python
@dataclass(frozen=True)
class DispatchDecision:
    kernel_id: str
    backend: Backend
    fallback_chain: Tuple[str, ...] = ()
    reason: str = ''
    reports: Tuple[CandidateReport, ...] = ()
    cache_hit: bool = False
    stack_id: str = ''
    stack_reports: Tuple[StackReport, ...] = ()
```

### _TuningResolution

调优数据库内部查询结果；record 是字典，source 标识 local/release，rejections 保存记录拒绝原因。

源码：[flashinfer/tuning/store.py:26](../../flashinfer/tuning/store.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class _TuningResolution:
    record: Optional[dict] = None
    source: str = ''
    rejections: tuple = ()
```

### SourceRef

源码的语言、包内路径和 SHA256 引用；构造记录本身不读源码或编译。

源码：[flashinfer/runtime/schema.py:378](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.source`、`JitSpec.source`、`BuildRecipe.source`、`BuildRecipe.sources`、`BuildRecipe.headers`、`BuildRecipe.host_sources`

```python
@dataclass(frozen=True)
class SourceRef:
    language: str
    locator: str
    digest: str
    schema_version: int = 1
```

### ArtifactRef

已交付产物的路径、摘要、格式、SoC 与构建身份；加载器校验实际字节与声明一致。

源码：[flashinfer/runtime/schema.py:297](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.artifact`、`JitSpec.source_artifacts`

```python
@dataclass(frozen=True)
class ArtifactRef:
    kind: ArtifactKind
    format: ArtifactFormat
    locator: str
    digest: str
    target_soc: str
    build_id: str
    size_bytes: Optional[int] = None
    schema_version: int = 1
    entry_point: Optional[str] = None
```

### WorkspaceFormula

规划接口用工作负载求 workspace 上界的公式；当前 Matmul 实际 workspace=0 由 host tiling 给出。

源码：[flashinfer/runtime/schema.py:203](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.workspace`、`KernelDescriptor.int_workspace`

```python
@dataclass(frozen=True)
class WorkspaceFormula:
    constant_bytes: int = 0
    dynamic_coefficients: Tuple[int, ...] = ()
    alignment: int = 32
```

### KernelLaunchABI

声明 host 入口、参数次序、可变参数和 stream 参数位置；连接 descriptor 与构建/加载契约。

源码：[flashinfer/runtime/schema.py:407](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.launch_abi`

```python
@dataclass(frozen=True)
class KernelLaunchABI:
    abi_name: str
    entry_point: str
    argument_names: Tuple[str, ...]
    mutable_arguments: Tuple[str, ...] = ()
    stream_argument: str = 'stream'
    pointer_width_bits: int = 64
    metadata_schema_version: int = 1
    schema_version: int = 1
```

### KernelBinaryABI

描述二进制参数、tiling 与错误协议；属于交付/校验元数据，不是每次选择时重新运行的 kernel。

源码：[flashinfer/runtime/schema.py:621](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.binary_abi`

```python
@dataclass(frozen=True)
class KernelBinaryABI:
    abi_name: str
    arguments: Tuple[KernelArgumentABI, ...]
    error_abi: KernelErrorABI
    calling_convention: str = 'c'
    return_primitive: CPrimitive = CPrimitive.I32
    pointer_width_bits: int = 64
    schema_version: int = 1
```

### KernelConstraints

WorkloadSpec/DeviceCapability 规划选择路径的额外约束；与 Tensor dispatcher 的 CapabilitySpec 路径区分。

源码：[flashinfer/runtime/schema.py:225](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.constraints`

```python
@dataclass(frozen=True)
class KernelConstraints:
    supported_socs: Tuple[str, ...] = ()
    dtype_signatures: Tuple[Tuple[str, ...], ...] = ()
    layout_signatures: Tuple[Tuple[str, ...], ...] = ()
    required_features: Tuple[str, ...] = ()
    quant_storage_dtypes: Tuple[str, ...] = ()
    deterministic: Optional[bool] = None
```

### KernelCapabilityBinding

声明能力目录的 domain/profile/rule 与指纹身份；属于 descriptor 元数据和校验关系。

源码：[flashinfer/runtime/schema.py:266](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.capability_binding`

```python
@dataclass(frozen=True)
class KernelCapabilityBinding:
    domain: str
    profile_id: str
    rule_id: str
    profile_fingerprint: str
    schema_version: int = 1
```

### GraphReplaySpec

图执行声明与验证边界；当前 Matmul 保持 NOT_CAPTURABLE，不能从存在该类型推断可入图。

源码：[flashinfer/runtime/schema.py:851](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.graph_replay`

```python
@dataclass(frozen=True)
class GraphReplaySpec:
    validated_kernel_id: str
    validation_tests: Tuple[str, ...]
    validated_environment: str
    strategy: str = 'host_kwargs'
    parameter_sources: Tuple[Tuple[str, str], ...] = ()
    required_arguments: Tuple[str, ...] = ()
    fixed_arguments: Tuple[Tuple[str, Any], ...] = ()
    tensor_metadata: Tuple[Tuple[str, Tuple[int, ...], Tuple[str, ...]], ...] = ()
    parameter_ranges: Tuple[Tuple[str, int, int], ...] = ()
```

### JitSpec

编译声明/runner 创建的编译单元身份；AOT 匹配和 JIT build/load 消费它，不等于一次调用的 InputSignature。

源码：[flashinfer/jit/core.py:57](../../flashinfer/jit/core.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class JitSpec:
    name: str
    domain: str
    generator_id: str
    generator_version: str
    target_soc: str
    environment_fingerprint: str
    specialization_fingerprint: str
    input_fingerprints: Tuple[Tuple[str, str], ...] = ()
    source_artifacts: Tuple[ArtifactRef, ...] = ()
    compile_options: Tuple[str, ...] = ()
    entry_points: Tuple[str, ...] = ()
    schema_version: int = JIT_SPEC_SCHEMA_VERSION
    kernel_id: Optional[str] = None
    source: Optional[SourceRef] = None
    output_path: Optional[str] = None
```

### BuildRecipe

算子构建与 Tensor/host ABI 配方；冻结、编译、AOT runner 和 Tensor adapter 使用它。此处已展开继承字段。

源码：[flashinfer/jit/recipe.py:369](../../flashinfer/jit/recipe.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class BuildRecipe:
    recipe_id: str
    version: int
    kernel_id: str
    implementation: str
    target_soc: str
    npu_arch: str
    source: SourceRef
    sources: Tuple[SourceRef, ...]
    headers: Tuple[SourceRef, ...]
    include_dirs: Tuple[str, ...]
    dependencies: Tuple[BuildDependency, ...]
    definitions: Tuple[str, ...]
    compile_options: Tuple[str, ...]
    link_options: Tuple[str, ...]
    device_entry: str
    host_entry: str
    launch_abi: str
    tiling_mode: str
    tiling_schema_version: int
    block_dim: int
    workspace_bytes: Union[int, str]
    task_type: str
    sync_policy: str
    toolchain: str = 'ascendc'
    output_kind: str = 'op_package'
    output_adapter: str = 'ascendc.op_package.v1'
    schema_version: int = 2
    host_sources: Tuple[SourceRef, ...] = ()
    package_vendor: Optional[str] = None
    package_op_type: Optional[str] = None
    tensor_parameters: Tuple[TensorParameter, ...] = ()
    shape_constraints: Tuple[ShapeConstraint, ...] = ()
    tiling_host_entry: Optional[str] = None
    workspace_host_entry: Optional[str] = None
    dtype_signatures: Tuple[Tuple[str, ...], ...] = ()
    allow_jit_fallback: bool = True
```

### RuntimeProvenance

AotKernelRunner 绑定执行资源时记录实际 aot/jit 来源；不改变 descriptor 的静态 compile_mode。

源码：[flashinfer/jit/runtime.py:27](../../flashinfer/jit/runtime.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class RuntimeProvenance:
    source: str
    directory: Path
    spec_fingerprint: str
    fallback_reason: Optional[str] = None
```

### JitLoadedModule

loader 返回校验凭据和不透明执行对象；runner 使用 opaque_module 执行。

源码：[flashinfer/jit/loading.py:172](../../flashinfer/jit/loading.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class JitLoadedModule:
    receipt: JitModuleLoadReceipt
    opaque_module: Any = field(repr=False, compare=False)
```

### WorkloadSpec

显式 manifest/规划接口的工作负载模型；build_plan/registry.select 使用，不替代本例 Tensor signature。

源码：[flashinfer/runtime/schema.py:155](../../flashinfer/runtime/schema.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class WorkloadSpec:
    op: str
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    static_dims: Tuple[int, ...] = ()
    dynamic_bounds: Tuple[int, ...] = ()
    attributes: Tuple[Tuple[str, str], ...] = ()
    quant_specs: Tuple[QuantSpec, ...] = ()
    causal: bool = False
    pos_encoding: str = 'NONE'
    deterministic: bool = False
    schema_version: int = 1
```

### DeviceCapability

显式规划接口的设备能力模型；支持版本、features 等，不等于 HardwareSignature 的三个字段。

源码：[flashinfer/runtime/schema.py:187](../../flashinfer/runtime/schema.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class DeviceCapability:
    soc_version: str
    soc_revision: str = ''
    ai_core_count: int = 0
    supported_dtypes: Tuple[str, ...] = ()
    features: Tuple[str, ...] = ()
    cann_version: str = ''
    torch_npu_version: str = ''
    compiler_version: str = ''
```

### Plan

wrapper 可持有 signature/hardware/decision/workspace；普通 Matmul facade 直接使用 decision，不构造 Plan。

源码：[flashinfer/runtime/plan.py:18](../../flashinfer/runtime/plan.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class Plan:
    signature: InputSignature
    hardware: HardwareSignature
    decision: DispatchDecision
    workspace_bytes: int = 0
```

### _WorkloadPlan

build_plan 返回的内部规划对象；绑定 workload/capability/kernel 和两类 workspace 上界。

源码：[flashinfer/runtime/plan.py:34](../../flashinfer/runtime/plan.py)。

被引用位置：作为函数输入/输出或容器成员使用，见上方用途与源码。

```python
@dataclass(frozen=True)
class _WorkloadPlan:
    workload: object
    capability: object
    kernel: object
    workspace_bytes: int
    int_workspace_bytes: int
```

### KernelArgumentABI

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/runtime/schema.py:566](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelBinaryABI.arguments`

```python
@dataclass(frozen=True)
class KernelArgumentABI:
    name: str
    passing: KernelArgumentPassing
    direction: KernelArgumentDirection
    primitive: CPrimitive = CPrimitive.U64
    nullable: bool = False
    required_alignment: int = 1
    pointee_abi_name: Optional[str] = None
    pointee_abi_fingerprint: Optional[str] = None
```

### KernelErrorABI

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/runtime/schema.py:601](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelBinaryABI.error_abi`

```python
@dataclass(frozen=True)
class KernelErrorABI:
    name: str
    codes: Tuple[KernelErrorCodeABI, ...]
    schema_version: int = 1
```

### BuildDependency

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/jit/recipe.py:46](../../flashinfer/jit/recipe.py)。

被引用位置：`BuildRecipe.dependencies`

```python
@dataclass(frozen=True)
class BuildDependency:
    name: str
    revision: str
    include_dir: str = 'include'
```

### TensorParameter

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/jit/recipe.py:62](../../flashinfer/jit/recipe.py)。

被引用位置：`BuildRecipe.tensor_parameters`

```python
@dataclass(frozen=True)
class TensorParameter:
    name: str
    role: str
    dtype: str
    shape: Tuple[Union[str, int], ...]
```

### ShapeConstraint

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/jit/recipe.py:86](../../flashinfer/jit/recipe.py)。

被引用位置：`BuildRecipe.shape_constraints`

```python
@dataclass(frozen=True)
class ShapeConstraint:
    name: str
    minimum: int
    maximum: int
    multiple_of: int = 1
```

### JitModuleLoadReceipt

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/jit/loading.py:75](../../flashinfer/jit/loading.py)。

被引用位置：`JitLoadedModule.receipt`

```python
@dataclass(frozen=True)
class JitModuleLoadReceipt:
    spec_name: str
    spec_fingerprint: str
    cache_record_fingerprint: str
    artifact_verification_fingerprint: str
    artifact_fingerprint: str
    loader_id: str
    loader_version: str
    module_token: str
    load_generation: int
    symbols: Tuple[JitResolvedSymbol, ...]
    schema_version: int = JIT_MODULE_LOAD_VERSION
```

### QuantSpec

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/runtime/schema.py:117](../../flashinfer/runtime/schema.py)。

被引用位置：`WorkloadSpec.quant_specs`

```python
@dataclass(frozen=True)
class QuantSpec:
    scheme: str
    storage_dtype: str
    compute_dtype: str
    accumulator_dtype: str
    scale_dtype: str = 'float32'
    granularity: str = 'tensor'
    group_size: Optional[Tuple[int, ...]] = None
    axis: Optional[Tuple[int, ...]] = None
    has_zero_point: bool = False
    physical_layout: str = 'logical'
    packing_order: Optional[str] = None
    schema_version: int = 1
```

### KernelErrorCodeABI

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/runtime/schema.py:593](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelErrorABI.codes`

```python
@dataclass(frozen=True)
class KernelErrorCodeABI:
    name: str
    code: int
    retryable: bool = False
    asynchronous: bool = False
```

### JitResolvedSymbol

该记录作为其他结构的字段使用，关联位置如下；具体校验与行为见源码。

源码：[flashinfer/jit/loading.py:47](../../flashinfer/jit/loading.py)。

被引用位置：`JitModuleLoadReceipt.symbols`

```python
@dataclass(frozen=True)
class JitResolvedSymbol:
    name: str
    symbol_token: str
    schema_version: int = JIT_MODULE_LOAD_VERSION
```

## 引用的枚举

### Backend

源码：[flashinfer/runtime/schema.py:26](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.backend`、`CandidateReport.backend`、`DispatchDecision.backend`

```python
class Backend(str, Enum):
    CANN = 'cann'
    ASCENDC = 'ascendc'
    TRITON = 'triton'
    EXTERNAL = 'external'
    CUSTOM = 'custom'
    REFERENCE = 'reference'
    ASCENDC_AOT = 'ascendc_aot'
    ASCENDC_JIT = 'ascendc_jit'
    ACLNN = 'aclnn'
```

### CompileMode

源码：[flashinfer/runtime/schema.py:46](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.compile_mode`

```python
class CompileMode(str, Enum):
    AOT = 'aot'
    JIT = 'jit'
    NONE = 'none'
```

### GraphCapability

源码：[flashinfer/runtime/schema.py:54](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelDescriptor.graph_capability`

```python
class GraphCapability(str, Enum):
    CAPTURE_SAFE = 'capture_safe'
    NEEDS_PARAM_UPDATE = 'needs_param_update'
    NOT_CAPTURABLE = 'not_capturable'
```

### ArtifactKind

源码：[flashinfer/runtime/schema.py:281](../../flashinfer/runtime/schema.py)。

被引用位置：`ArtifactRef.kind`

```python
class ArtifactKind(str, Enum):
    FILE = 'file'
    JIT_SOURCE = 'jit_source'
    BUILTIN = 'builtin'
```

### ArtifactFormat

源码：[flashinfer/runtime/schema.py:287](../../flashinfer/runtime/schema.py)。

被引用位置：`ArtifactRef.format`

```python
class ArtifactFormat(str, Enum):
    ASCENDC_OBJECT = 'ascendc_object'
    ASCENDC_OP_PACKAGE = 'ascendc_op_package'
    SHARED_LIBRARY = 'shared_library'
    TRITON_NPUBIN = 'triton_npubin'
    ASCENDC_SOURCE = 'ascendc_source'
    ACLNN_BUILTIN = 'aclnn_builtin'
```

### CPrimitive

源码：[flashinfer/runtime/schema.py:432](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelBinaryABI.return_primitive`、`KernelArgumentABI.primitive`

```python
class CPrimitive(str, Enum):
    U8 = 'u8'
    U16 = 'u16'
    U32 = 'u32'
    U64 = 'u64'
    I8 = 'i8'
    I16 = 'i16'
    I32 = 'i32'
    I64 = 'i64'
    F32 = 'f32'
    F64 = 'f64'
```

### KernelArgumentPassing

源码：[flashinfer/runtime/schema.py:553](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelArgumentABI.passing`

```python
class KernelArgumentPassing(str, Enum):
    POINTER = 'pointer'
    VALUE = 'value'
    OPAQUE_HANDLE = 'opaque_handle'
```

### KernelArgumentDirection

源码：[flashinfer/runtime/schema.py:559](../../flashinfer/runtime/schema.py)。

被引用位置：`KernelArgumentABI.direction`

```python
class KernelArgumentDirection(str, Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    INOUT = 'inout'
```
