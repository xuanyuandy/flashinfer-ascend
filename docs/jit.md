# JIT 声明、惰性编译与装载（T18A / T18B / T13-I）

`flashinfer.jit` 的编译单元沿用 `JitSpec`。声明、登记、选型和查询仍无需
Triton、torch 或 CANN 编译器。只有显式调用 `build_and_load()` 才进入
编译 / 装载路径。Triton 与 Ascend C host_shared 均已在 A5 上完成真实编译和执行。
Ascend C 的工具链发现与配方解析也是显式操作，不由 import、auto 或纯声明触发。
JIT 主产物为 host `.so`；AOT 的标准 op-package 留给 T13-II。

## A5 实测产物先于装载设计

2026-10-22，Python 3.11、Triton 3.6.0、Ascend950PR_9579，最小 `add_one`：

| 文件 | 实际格式 / 用途 |
|---|---|
| `add_one.source` | 文本，带源码位置的初始 IR，并非原 Python 文件 |
| `add_one.ttir` | 文本，Triton IR |
| `add_one.ttadapter` | 文本，Ascend 适配 IR |
| `add_one.mlirbc` | MLIR bytecode，开头 `ML ef 52` |
| `add_one.bcmlir` | 文本，解码 / 转换后的 MLIR |
| `add_one.npubin` | ELF64 little endian，EXEC，NPU machine `0x1029` |
| `add_one.json` | JSON，目标、kernel_name、shared、mix_mode、编译选项等 |
| `__grp__add_one.json` | JSON，Triton 的缓存文件集合索引（含绝对路径） |
| `npu_utils.so` | x86-64 ELF shared object，Python `PyInit_npu_utils` 扩展 |
| `launcher_cxx11abi1.cpython-311-x86_64-linux-gnu.so` | x86-64 Python `PyInit___triton_launcher` 扩展 |
| `lock/<hash>.lock` | launcher 编译锁，0 字节 |

设备 ELF 的已定义 GLOBAL FUNC 是 `add_one`，元数据 `name/kernel_name` 也为
`add_one`。两个 CPU `.so` 是运行时辅助扩展，不导出可用 ctypes 调用的设备
kernel。最小 kernel 已执行、误差为 0；调用方无需额外 device linking，
工具链已通过 `-cce-link-aicore-ll-module .../libdevice.10.bc` 处理 libdevice。
这只证明本次独立 kernel 的链路，不承诺任意跨模块设备符号链接。

因此新增 `ArtifactFormat.TRITON_NPUBIN`，不将 NPU ELF 标成 shared_library。
详细原始清单、命令与验收结果见 [T18B 实测报告](reports/T18B-report.md)。

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
没有新增与 `SourceRef` / `ArtifactRef` 平行的溯源模型。声明输出可使用 `.bin`
等任意后缀；格式由验证后的实际字节确定，Triton 原生文件名另行保存供其加载器使用。

旧 Attention 配方继续使用现有 `source_artifacts`，三个新字段不写入其 JSON，
原有指纹不变。新编译单元使用 `SourceRef`，不混用旧的源码产物字段。
`source_materialized` 表示已登记显式源码身份，不验证文件是否存在。

原有 `build_cache_key()` 仍只覆盖输入签名、硬件与工具链。新单元的完整构建
身份是 `spec.fingerprint`；T18B 的磁盘缓存需要使用这一完整身份，不能只用
旧 helper 的结果，否则会遗漏 kernel / source / options 等维度。T18B 磁盘路径、
锁和 manifest 全部使用 `spec.fingerprint`；旧 helper 只保留兼容用途。

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
`compile` 的正常返回只表示编译器调用结束，统一层仍必须验证输出。Triton
与 Ascend C 均提供真实实现。编译器只能向统一层提供的 staging 根写出声明产物，不能
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

默认查询全局 KernelRegistry / JitSpecRegistry / `jit_cache_index`；`cache` 可提供
独立的 `JitCacheIndex` 内存记录。函数不生成声明、不探测文件、不调用任何编译器、
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
统一入口完成编译验证后即登记身份（装载失败也如实反映已编译）；`artifact_path`
报告所登记的 generation 内绝对路径。旧的纯内存 fixture 无实际路径时仍返回
`ArtifactRef.locator`。查询不扫描磁盘；新进程先通过显式构建 / cache-only 入口
恢复记录后再查询，不因磁盘上恰好存在同名文件而宣称成功。

## 显式编译与装载

```python
from flashinfer.jit import DiskJitBuildStore, JitCacheIndex, build_and_load

# spec 来自 declare_kernel；必须提供真实 SourceRef 摘要与显式构建身份。
store = DiskJitBuildStore()  # 只构造配置，不访问磁盘
cache = JitCacheIndex()
module = build_and_load(spec, backend=backend, store=store, cache=cache)
module.opaque_module[(rows,)](x, residual, weight, hidden, eps)
```

可直接运行的真实声明见 `examples/jit_a5.py`，复用 `_norm_kernels.py` 的
Gemma fused RMSNorm。没有修改内置 79 条 descriptor、capability 或 dispatch，
也没有自动给旧 Attention 配方赋予可执行 ABI。

`TritonBackend(source_root=..., python_executable=...)` 默认使用包目录和当前
Python。源码为单个独立的、可信的 Python 模块，声明的入口必须是该模块中的
`@triton.jit` 函数，且实际导出的 `name/kernel_name` 和设备符号必须相同。
源码读取、SHA-256 校验和快照仅发生在显式编译时，声明时不导入源码。
源码的额外文件依赖尚无打包支持，不得将未指纹化的外部源码当作同一构建身份。

Triton `compile_options` 采用三个显式 JSON 对象字符串：

```python
compile_options = (
    'signature={"x_ptr":"*fp32","r_ptr":"*fp32","w_ptr":"*fp32","hidden":"i32","eps":"fp32"}',
    'constexprs={"BLOCK":128}',
    'options={"num_warps":32}',
)
```

`signature` 必填，另外两项默认空对象。runtime signature 与 constexprs 必须
完整且不重叠地覆盖函数参数；constexprs 必须精确匹配 `tl.constexpr`。
launcher 按函数形参顺序生成，JSON key 的顺序不会改变参数 ABI。
不把 `-O3` 等字符串猜成另一套参数语义，不推断 dtype / shape / constexpr。
声明阶段仍保留原始选项顺序；不支持的选项在编译边界明确报错。
Triton 生成器版本为 3，区别于 18A 的纯声明生成器版本 1。

worker 在私有 `TRITON_CACHE_DIR` 调 `triton.compile(ASTSource(...))`，并预编译
两个 CPU runtime 扩展、解析设备入口，不执行 kernel。父进程校验 ELF 类型、
machine、已定义符号及元数据，再以 `ArtifactRef` 记录声明位置的 NPU ELF。
原生 metadata group 保留相对路径，丢弃 Triton 含 staging 绝对路径的 group 索引。
快照通过固定的源码内容路径与 worker 内 linecache 提供给 `@triton.jit`，避免随机
staging 路径进入 IR。另经重复实编确认 `.mlirbc` 字节不稳定，而文本 IR
和 npubin 相同；它不是装载输入，故编译结束后丢弃该中间文件。其余持久文件全部
进入 manifest，重建保持不可变 `JitCacheIndex` 的原有冲突检查，不放宽身份语义。

`DefaultJitModuleLoader` 实现原有 `JitModuleLoader` 协议：Triton 分支恢复
`CompiledKernel`，通过已编译的 `npu_utils.load_kernel_binary` 和 Python launcher
装载 / 执行，绝不再次编译设备 kernel 或 launcher。无需重新导入原 kernel 源码。
CPU `.so` 使用 Python extension loader；只有显式 shared_library 走 `ctypes.CDLL`。
实际缓存复用已在新进程禁止所有 subprocess 的条件下验证。
Triton 适配器依赖 3.6 的运行时 ABI，检查 triton/torch/torch_npu/Python ABI 和实际设备；
不同版本明确报错，需要重新编译 / 适配。加载器返回带精确符号的现有 receipt，
并按记录与 NPU device 保留进程内 LRU。调用方持有模块后，launch 不查磁盘 / 编译。

## 缓存、锁和原子发布

`JitBuildStore` 的四个钩子及 `JitCacheRecord.validate_spec` 语义保持不变。
`DiskJitBuildStore` 是其 POSIX 实现，统一入口使用它作为本地字节来源：

```text
${FLASHINFER_NPU_CACHE_DIR:-~/.cache/flashinfer}/jit/
  .locks/<完整 spec.fingerprint>.lock
  .staging/<key>-<私有随机后缀>/
  .objects/<key>-<generation>/
    <spec.output_path>          # 声明位置的真实 NPU ELF / CPU .so
    triton.json                # 相对 native group、launch ABI、版本
    triton-cache/...           # 原生 IR / JSON / npubin / runtime .so
    source.py, request.json, build.log
    manifest.json              # 完整 spec、record、文件 size / SHA-256
  <key[:2]>/<完整 key> -> ../../.objects/<key>-<generation>
  .logs/<key>-...log            # 失败后仍可定位的日志
```

查询 → `flock` → 锁下二次查询 → 私有 staging 编译 → 验证 → 写完整 manifest →
fsync → rename 到不可变 generation → 原子 `os.replace` 发布 generation 指针。
锁文件永不随目的目录替换或删除，两个进程 / 线程只会有一个编译者。
manifest 和全部产物先完成，再发布同一个指针，读者看到完整代或未命中。
损坏缓存按 18A 合约视为 miss，在锁下重建；加载前重新核对 spec / 文件集合 /
所有文件的 size、digest，拒绝 symlink 输出及路径逃逸。
旧 generation 保留，保证正在读取旧路径的进程不会因发布而丢失文件。
失败 staging 清理，编译日志移到 `.logs`，异常里的日志位置同步更新。

默认包目录只读；缓存、staging、产物分离。用户 kernel 可显式使用
`DiskJitBuildStore(root, namespace="company")`，物理落到 `jit/user/company/`。
该参数不自动推断，接入者应为用户源码指定隔离 store。缓存淘汰 / 旧 generation
清理不在本次实现中。只使用支持 `flock` 和原子 rename/symlink 替换的本地 POSIX 文件系统。

`JitCompilationPolicy.CACHE_ONLY/DISABLED` 禁止 miss 编译，仍可加载已验证的缓存。
默认策略读取 `FLASHINFER_NPU_JIT_DISABLE`，不影响纯 import / 声明 / 查询。

## 失败边界与待接入部分

`JitCompilationError` 给出 kernel、完整 spec、worker 命令、返回码、错误摘要和
持久日志。`JitLoadError` 表示字节 / 环境 / 动态装载失败，`JitSymbolError` 单独
表示入口未导出 / 未解析；二者保存 traceback 日志。Python API 装载没有进程返回码，
此时 `returncode=None`，不会虚构编译命令或成功产物。

Ascend C 使用 `bisheng` 和 CANN ASC CMake，host wrapper 与 device binary 一起链接为
host `ET_DYN`。不把 NPU relocatable `.o` 交给 CDLL。跨模块 device linking、运行时
ABI 跨版本适配，以及把旧 Attention 配方接到真实 kernel，仍需各自实现和验证。

## Ascend C BuildRecipe（T13-I）

`csrc/recipes/matmul_probe.json` 声明两个独立 kernel_id 的配方，语言均为 `ascendc`。
这两个是新增的固定 FP32 GEMM **测试记录**，示例只在独立 KernelRegistry 显式登记；
既有全局 79 条记录、选型、graph capability 与证据门禁均未改变。该 probe 不提供生产 GEMM 性能承诺。

`BuildRecipe` 的完整字段分组如下：

| 字段 | 作用 |
|---|---|
| `schema_version, recipe_id, version, kernel_id` | 配方及执行体身份 |
| `toolchain, implementation, target_soc, npu_arch` | 同一个 Ascend C 后端的 native/CATLASS 变体及精确目标 |
| `source, sources, headers, include_dirs` | 主 SourceRef、所有编译单元与本仓头文件的内容摘要；include 根只是定位 |
| `dependencies` | `BuildDependency(name, revision, include_dir)`；CATLASS 固定 commit，native 为空 |
| `definitions, compile_options, link_options, sync_policy` | 有序 flags 和同步约定，不能从全局 CPATH 注入依赖 |
| `device_entry, host_entry, launch_abi, task_type` | 设备入口、宿主 C 入口、ABI id 和 kernel 类型 |
| `tiling_mode, tiling_schema_version, block_dim, workspace_bytes` | tiling/launch 协议；v1 实现固定参数、无 buffer/workspace 的 AIC-only |
| `output_kind, output_adapter` | 本次只支持 `host_shared / ascendc.host_shared.v1`，其他组合明确失败 |

新增 kernel 须提供自己的 device 源码、host launcher、完整 source/header 摘要、上述配方，
以及 source/target/launch ABI/tiling 版本匹配的 KernelDescriptor。变体使用不同具体
kernel_id / SourceRef；配置只选择已经存在的变体，不修改已选 descriptor 或失败后换实现。
主 SourceRef 和额外文件都在编译前重新校验；仅显式列出的文件会被快照到编译目录，
因此遗漏本仓 include 会直接编译失败，不能悄悄读工作树中的未声明文件。

统一路径为：

```text
AscendCBuildConfig.select -> 选中 BuildRecipe
AscendCToolchain.discover + resolve_build_recipe -> 冻结工具链/实际依赖内容
AscendCBackend(recipes=(resolved,), toolchain=toolchain)
declare_kernel -> JitSpec -> build_and_load -> 原有 ELF verifier / CDLL loader
```

`resolve_build_recipe` 校验固定 Git commit，并散列实际 include 树，脏 header 也改变
身份；native 根本不读取 CATLASS 路径或运行 Git。只有选中 CATLASS 变体才需要手动
`git submodule update --init -- 3rdparty/catlass`，不由 Python 下载依赖。
`AscendCBuildConfig` 默认 native，支持 `overrides=((recipe_id, implementation), ...)`；
JSON 文件的 overrides 使用同名对象。找不到变体或出现歧义均明确报错。

配方摘要与解析结果摘要分别进入 `JitSpec.input_fingerprints` 的 `build_recipe` 和
`resolved_recipe`，同时保留 descriptor、包版本、环境、目标、特化、入口、flags、输出路径身份。
解析结果覆盖 CMake 输入和实际依赖内容；工具链身份覆盖 bisheng/CMake/host C++ 的版本与
binary、CANN 版本/构建配置、host ABI 和 driver。编译钩子只消费这份冻结结果，不再读取实现选择配置。

同一个 `csrc/CMakeLists.txt` / `cmake/HostShared.cmake` 消费 sources/includes/definitions；
没有 CATLASS 分支、独立编译器、缓存或 loader。CMake scratch（含临时头文件副本）在
编译结束后清理，只发布 `kernel.so`、`ascendc-recipe.json`、`build.log` 与原有 manifest。
锁、二次查询、staging、全文件校验、原子发布、失败日志和加载 receipt 全部复用 T18B。

重放本次 A5 验证（先配置 CANN 环境，Python 固定如下）：

```bash
PY=/root/miniforge3/envs/flashinfer-a5/bin/python
E=/root/.cache/flashinfer/t13-i-evidence
git submodule deinit -f -- 3rdparty/catlass
rmdir 3rdparty/catlass
test ! -e 3rdparty/catlass
env -u CPATH -u CPLUS_INCLUDE_PATH "$PY" examples/ascendc_jit_a5.py \
  --config examples/ascendc_native.json --cache-dir "$E/no-catlass"
git submodule update --init --no-fetch -- 3rdparty/catlass
"$PY" examples/ascendc_jit_a5.py \
  --config examples/ascendc_catlass.json --cache-dir "$E/no-catlass"
```

两份配置文件只有 `implementation` 不同，脚本及源码无需编辑。两边首次均输出
`disk_cache_hit: False`、`LOAD_OK: shared_library entry: fi_matmul_probe_v1`，
256 个输出逐项等于独立 CPU GEMM，`max_abs_error: 0.0`，前八值均为
`[-1.0, 9.0, -1.0, 4.0, -11.0, -1.0, 9.0, -1.0]`。输出预填 NaN，空 launch 不能通过检查。
生成器给设备符号使用各自唯一名字，宿主遵守同一个四指针 C ABI。

`--cache-only` 从示例保存的 JitSpec 重载；该模式禁止 subprocess/compile，既不发现
编译器，也不需要源码或 CATLASS。wheel/sdist 仅携带本仓自己的 CMake、配方、源码和
launcher；显式 `FLASHINFER_AOT_ALL=0` 可仅打包源码，普通 wheel 构建默认执行生产 AOT。安装后的源码位于
`flashinfer/data/build/csrc/`，可用 `--source-root` 显式覆盖其上级 `data/build/`，
CATLASS 可用 `--catlass-root` 指向调用者初始化的固定 Git checkout。

验证范围仅 CANN 9.2 beta.2 / A5 / FP32 16×16×16 / AIC-only / 固定 tiling，
尚未验证 A2/A3/310P、动态 shape、mixed task、workspace、性能或图捕获。
13-II 已确定 AOT 主产物为标准 op-package；实施前还需固定 package manifest、
完整多文件清单与 digest、vendor/版本/SOC 布局、ACLNN/tiling ABI、build-only 导出，
以及安装或搜索路径与同进程多版本隔离规则。不能用现有单文件 ArtifactRef 冒充包完成。
