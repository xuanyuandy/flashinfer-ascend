# JIT 声明与状态查询（T18A）

`flashinfer.jit` 的编译单元沿用 `JitSpec`。本阶段只构造元数据、登记声明和
查询映射；Triton / Ascend C 编译钩子均抛出说明等待 T18B 的
`NotImplementedError`。导入、登记、选型和查询无需 Triton、torch 或 CANN 编译器。

## 编译单元

新声明在原有配方字段之上增加三个字段：

```python
kernel_id: Optional[str] = None
source: Optional[SourceRef] = None
output_path: Optional[str] = None
```

三个字段必须一起提供；编译单元还必须有明确的 `target_soc` 和唯一的
`entry_points`。`compile_options` 保留调用者给出的顺序；源码内容摘要、编译
目标、选项、入口、期望路径、环境与特化指纹均进入声明身份。内置后端也将
descriptor 身份、包版本和生成器版本纳入身份。不从本机探测工具链，不自行
推测编译选项或特化参数。

`source` 直接使用 `KernelDescriptor.source` 的 `SourceRef`。`output_path`
是产物根内规范的相对 POSIX 路径；根目录由统一层在 T18B 解析。
它只是期望位置，不带虚构的 digest 或 size。编译结果经统一层验证后才能
生成现有的 `ArtifactRef`，其中的 locator、target_soc、entry_point 与声明对应。
本阶段没有新增与 `SourceRef` / `ArtifactRef` 平行的溯源模型。

旧 Attention 配方继续使用现有 `source_artifacts`，三个新字段不写入其 JSON，
原有指纹不变。新编译单元使用 `SourceRef`，不混用旧的源码产物字段。
`source_materialized` 表示已登记显式源码身份，不验证文件是否存在。

原有 `build_cache_key()` 仍只覆盖输入签名、硬件与工具链。新单元的完整构建
身份是 `spec.fingerprint`；T18B 的磁盘缓存需要使用这一完整身份，不能只用
旧 helper 的结果，否则会遗漏 kernel / source / options 等维度。

## 两个后端钩子

`JitBackend` 是可运行期检查的 Protocol：

```python
backend_id: str

def declare(
    descriptor: KernelDescriptor, *, target_soc: str, output_path: str,
    entry_point: str, environment_fingerprint: str,
    specialization_fingerprint: str, compile_options: Tuple[str, ...] = (),
) -> JitSpec: ...

def compile(spec: JitSpec, *, build_directory: Path) -> None: ...
```

新后端实现这两个方法和 `backend_id`，交给同一个 `declare_kernel()` 即可；
不需要在 dispatch 增加分支。声明阶段不读文件、不导入 kernel 源码、不编译。
`compile` 的正常返回只表示编译器调用结束，统一层仍必须验证输出；内置
占位实现始终抛错。编译器只能向统一层提供的 staging 根写出声明产物，不能
自行管理缓存、发布、装载或登记。

`TritonBackend` 接受 `language="triton"`；`AscendCBackend` 接受
`language="ascendc"`。二者也支持对应语言的 custom / external descriptor。
目标优先采用 descriptor 的产物目标或 `target_soc`；缺少时必须显式提供。
显式目标不得与 descriptor 冲突。目标未写入 descriptor 时，生成声明不改变
descriptor 或选型规则。

例如，下面只登记一个独立测试 kernel 和声明，不创建产物：

```python
from flashinfer.jit import (
    JitSpecRegistry, TritonBackend, declare_kernel, describe_artifact,
)
from flashinfer.runtime import Backend, KernelDescriptor, KernelRegistry, SourceRef

descriptor = KernelDescriptor(
    "triton/example/Ascend910B", "example", Backend.TRITON,
    source=SourceRef("triton", "kernels/example.py", "a" * 64),
    target_soc="Ascend910B",
)
registry = KernelRegistry((descriptor,))
specs = JitSpecRegistry()
spec = declare_kernel(
    descriptor.kernel_id, backend=TritonBackend(), registry=registry,
    spec_registry=specs, output_path="kernel.bin", entry_point="example_entry",
    environment_fingerprint="b" * 64, specialization_fingerprint="c" * 64,
)
rows = describe_artifact(descriptor.kernel_id, registry=registry, spec_registry=specs)
print([row.to_dict() for row in rows])
```

例中的摘要是说明数据；真实接入者须提供实际源码、编译环境和特化身份。
现有 79 条内置记录保持原样，包括尚未声明 SourceRef 的 Triton norm 记录。

## 查询语义

```python
describe_artifact(
    kernel_id: str, *, registry=None, spec_registry=None, cache=None,
) -> Tuple[JitArtifactStatus, ...]
```

默认查询全局 KernelRegistry / JitSpecRegistry；`cache` 可提供已有的
`JitCacheIndex` 内存记录。函数不生成声明、不探测文件、不调用任何编译器、
字节读取器或装载器。每个特化 / 环境声明各一行；descriptor 中没有绑定到
具体声明的产物单独列一行，避免根据相同路径猜测编译成功。

每行包含 kernel_id、known_kernel、backend、state、is_compiled、source、
target_soc、artifact、artifact_path、expected_artifact_path、entry_point、
compile_options、spec_name、spec_fingerprint 和 reason。SourceRef / ArtifactRef
仍是原来的 schema 对象；`to_dict()` 给出可 JSON 序列化的快照。

| state | is_compiled | 含义 |
|---|---|---|
| unknown_kernel | None | kernel_id 未登记 |
| undeclared | None | 已登记，但未声明源码 / 产物 |
| source_only | False | 只有源码身份，尚无编译单元 / 产物记录 |
| declared | False | 有编译单元，尚无与之精确绑定的产物记录 |
| artifact_recorded | True | descriptor 或精确绑定的缓存记录提供已编译 ArtifactRef |
| builtin | None | 内建提供者，不适用本地编译状态 |

这里的 `is_compiled` 只描述已登记的编译身份。`artifact_recorded` 不证明文件
仍存在、字节有效、ABI 可用或符号已解析；这些由 T18B 的读取、校验和装载
流程负责。期望路径不放进 `artifact_path`，编译声明不构造假的 ArtifactRef。

## T18B 的落点

`JitBuildStore` 只定义 `lookup / lock / staging / publish`。统一层应执行
查询 → 锁 → 锁下二次查询 → staging 编译 → 验证 → 产物和 manifest 原子发布。
锁的身份必须在目的目录被替换时仍然有效。协议没有安装文件锁、磁盘缓存、
原子 rename 或淘汰逻辑。现有 `JitModuleLoader` 继续承担装载协议；尚无默认装载器。

以下细节需要在真实编译中验证后再定：Triton 产物格式及元数据文件集合、编译参数
与特化的转换、device linking 要求、CANN 编译 / 链接命令与导出包装、原生符号到加载器及 launch ABI
的绑定、构建日志 / 错误结果结构。当前声明中的入口是调用者的期望入口，
没有保证它已被某个真实编译器导出。不能据此假设 Triton 产物是可用 ctypes
装载的共享库。
