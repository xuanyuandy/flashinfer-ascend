# T18B · A5 实编、装载与磁盘缓存验收

日期：2026-10-22。只保留工作区修改，未 commit / push；未修改 `docs/tasks/`、
`docs/TASKS.md`、dispatch 过滤排序、capability 谓词或 79 条内置 descriptor。

## 先实测 Triton 产物，再确定装载方案

环境：`/root/miniforge3/envs/flashinfer-a5/bin/python`，Python 3.11，
torch `2.10.0+cpu`，torch_npu `2.10.0.post7`，Triton `3.6.0`，
CANN `/usr/local/Ascend/cann-9.2.0-beta.2`，8 张 NPU，实际设备 `Ascend950PR_9579`。

第一次设计前实测命令：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python /tmp/flashinfer-t18b-probe/probe.py
```

该独立程序编译 `add_one(X,Y,N,BLOCK)` 并真实执行 `arange(257)+1`。
初次目录中：设备编译缓存目录 `3XWGBD6OCHHTI54GVSGQYD5RBM6CFU3QQTTUWHGAVM3BXKFXFE4A`，
`npu_utils` 目录 `QYJQQRL4G6KXBC7GBZIXCJ4AJI`，launcher 目录
`ZMXN3PCZ7XIHSMJHX3KBGWGE36PIXEXGGIP6KO4SG5TKOBXOI3OQ`。

| 文件（相对上述目录） | 字节 | 格式 / 用途 |
|---|---:|---|
| `add_one.source` | 3937 | 文本初始 IR，带源码位置，非 Python 原文件 |
| `add_one.ttir` | 2261 | Triton 文本 IR |
| `add_one.ttadapter` | 2471 | Ascend 适配文本 IR |
| `add_one.mlirbc` | 1634 | MLIR bytecode，magic `4d 4c ef 52` |
| `add_one.bcmlir` | 2453 | 文本 MLIR |
| `add_one.npubin` | 20496 | ELF64 LE，EXEC，NPU machine `0x1029` |
| `add_one.json` | 4043 | 目标、选项、kernel_name、shared、mix_mode 等元数据 |
| `__grp__add_one.json` | 918 | JSON 文件集合索引，原生索引含绝对路径 |
| `npu_utils.so` | 500784 | x86-64 CPU ELF shared object，Python 扩展 |
| `launcher_cxx11abi1.cpython-311-x86_64-linux-gnu.so` | 255088 | x86-64 CPU Python launcher 扩展 |
| `lock/cb2eddbc59fdd0793127bed41358c4df9e8b92e6321fe53b923766a706ee46dd.lock` | 0 | launcher 编译锁 |

此外 `/tmp/flashinfer-t18b-probe` 有输入 `probe.py` 和输出 `probe.log`；它们不是
Triton 编译产物。没有发现可直接 ctypes 调用 `add_one` 的 CPU kernel `.so`。

`readelf -h -W -s add_one.npubin` 的关键输出：

```text
Class: ELF64
Type: EXEC (Executable file)
Machine: <unknown>: 0x1029
567: 0000000000000180 632 FUNC GLOBAL DEFAULT 1 add_one
568: 0000000000000000   4 OBJECT WEAK DEFAULT 2 add_one__
```

元数据 `name=kernel_name=add_one`，Triton 实际装载后的 module / function 均非空。
CPU 辅助扩展分别导出 `PyInit_npu_utils`、`PyInit___triton_launcher`。
设备 kernel 由 `npu_utils.load_kernel_binary` 与 launcher 运行，不是 CPU dlsym 函数。

实际输出：

```text
DEVICE Ascend950PR_9579
TARGET GPUTarget(backend='npu', arch='Ascend950PR_9579', warp_size=0)
NUMERICS True max_abs_error 0.0
```

本次 kernel 不需要调用方执行额外 device linking。元数据 `bisheng_options` 中实际包含
`-cce-link-aicore-ll-module .../triton/backends/ascend/lib/libdevice.10.bc`，由工具链处理。
该事实仅覆盖本次单 kernel；不扩展为任意跨模块设备符号链接的承诺。

仓内可复现命令（重新列出整个产出目录、magic、ELF header / 入口）：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b-acceptance --probe
```

再次得到同样的文件种类和 `PASS add_one max_abs_error= 0.0`。原 Python 文件路径
不同会改变带位置的 IR 大小及缓存 hash；不能将第一次字节数当成所有构建的固定值。

## Ascend C 工具链

实测 `shutil.which`：

```text
bisheng /usr/local/Ascend/cann-9.2.0-beta.2/bin/bisheng
ccec    /usr/local/Ascend/cann-9.2.0-beta.2/bin/ccec
ascendc None
```

`bisheng --version` 与 `ccec --version` 均成功，均输出：

```text
2026-08-18T19:12:48+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/local/Ascend/cann-9.2.0-beta.2/bin
```

实际设备编译：

```cpp
extern "C" __global__ __aicore__ void t18b_ascendc_probe(__gm__ float *x) { x[0] = 1.0f; }
```

```bash
bisheng -xcce --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 -c /tmp/flashinfer-t18b-probe/minimal.cpp -o /tmp/flashinfer-t18b-probe/minimal.o
```

```text
ASCENDC_RETURN_CODE=0
Type: REL (Relocatable file)
Machine: <unknown>: 0x1029
11: 0000000000000000 104 FUNC GLOBAL DEFAULT 2 t18b_ascendc_probe
```

结论是工具链存在且可以编译设备代码。此次尚未集成并执行验证 Ascend C host wrapper
和 launch ABI，因此 `AscendCBackend.compile` 继续显式 `NotImplementedError`，不将
设备 object 冒充可执行 host shared library；也没有声称两个原生后端都已完成。

## 实现与边界

Triton 选择独立 subprocess 中调用 Python `ASTSource/compile`；父进程复用 Triton
`CompiledKernel`、已编译 `npu_utils` 和 launcher 装载 NPU ELF。装载不导入原 kernel
源码、不发现编译器、不重编译 launcher。预编译 CPU `.so` 通过 Python 扩展机制载入；
显式 host shared_library 才使用 `ctypes.CDLL`，真实 CPU 库测试验证此分支。

声明参数只解析显式 `signature=<JSON>`、`constexprs=<JSON>`、`options=<JSON>`，
不推断 dtype / shape / 常量 / CLI 编译选项。实际 launcher 必须按源函数形参顺序生成，
JSON key 的排序不影响 ABI。Triton 生成器版本为 3；旧 Attention 配方序列化不变。

四钩子协议未改。`DiskJitBuildStore` 用完整 `spec.fingerprint` 分目录与加锁；锁在
`.locks/`，永不删除 / 随发布替换。统一层锁下二次查缓存；私有 `.staging/` 完成编译，
验证 ELF / 入口 / 元数据后生成完整 manifest，记录所有持久产物的 size / SHA-256。
文件与目录 fsync 后 rename 到不可变 `.objects/<key>-<generation>`，最后原子替换
`<key[:2]>/<key>` symlink。manifest 和产物在同一个 generation 中同时可见。
旧 generation 保留，不影响已经拿到旧路径的读者；损坏重建不松动同名记录冲突检查。

重复 A5 实编发现：即使文本 IR 和 npubin 相同，`.mlirbc` 字节仍会变化。固定 snapshot
源码位置后此差异仍存在。最终实现只在编译期使用该中间字节码，结束后丢弃；保留的
源码、文本 IR、npubin、JSON 和 CPU 扩展仍全部进入 manifest。这样两次重建的现有
`JitCacheRecord` 身份相同，无需放宽 `JitCacheIndex.publish` 的不可变语义。

失败 staging 清理，日志移入 `.logs` 并更新异常路径；编译、装载、缺符号分别报告
`JitCompilationError`、`JitLoadError`、`JitSymbolError`，带 kernel / spec、命令、
返回码及持久日志。非 subprocess 的装载步骤用 `returncode=None`，不虚构命令。

`describe_artifact` 仍只读内存，不扫描磁盘。编译验证后登记真实 `ArtifactRef` 和实际
generation 路径，装载失败也不会假称没有编译。全局查询默认消费 `jit_cache_index`；
独立测试 / 接入者可提供自己的 registry、spec registry 和 index。现有 norm runner
未自动改路；验收使用同一 `_norm_kernels.py` 的独立 T18B 声明，避免修改 79 条静态记录。

## 改动文件

| 文件 | 改动与原因 |
|---|---|
| `flashinfer/jit/backends.py` | Triton 真实 subprocess 编译、源码摘要核对、显式选项；Ascend C 说明缺少执行 ABI |
| `flashinfer/jit/_triton_compile.py` | 独立 worker，真实设备编译、launcher 预编译、相对元数据、稳定源码位置 |
| `flashinfer/jit/build.py` | 统一惰性入口、锁下二次检查、验证、发布、编译身份登记与装载 |
| `flashinfer/jit/cache.py` | 四钩子磁盘实现；稳定 flock、私有 staging、文件校验、原子代发布与失败日志保留 |
| `flashinfer/jit/loading.py` | 默认加载器、Triton runtime ABI 还原、预编译 Python 扩展、host ctypes、LRU 与精确符号 receipt |
| `flashinfer/jit/_elf.py` | 不依赖 binutils 的 ELF header / 已定义函数入口检查；host 入口须在 dynsym |
| `flashinfer/jit/errors.py` | 区分编译、装载和缺符号错误，保留定位字段 |
| `flashinfer/jit/status.py` | 默认消费构建内存登记，报告真实路径，继续禁止查询时 I/O |
| `flashinfer/jit/cache_key.py` | 标明旧 helper 不再作为构建缓存身份，兼容原函数 |
| `flashinfer/jit/__init__.py` | 导出显式统一入口及默认实现，导入时不构建 / 探测 |
| `flashinfer/runtime/schema.py` | 增加真实格式 `TRITON_NPUBIN`，不冒充 shared_library |
| `examples/jit_a5.py` | 可复现 raw probe、CPU 参考数值验证、禁止 subprocess 的新进程加载、同 index 损坏重建 |
| `tests/test_jit_build.py` | 两进程 / 两线程、原子发布、损坏、策略、完整 key、失败日志、装载 / 缺符号验证 |
| `tests/test_jit_trigger_layer.py` | 占位断言更新为真实错误边界；强化无依赖 / 无编译器审计与账本检查 |
| `tests/test_checkpoint_033_attention_jit_module_loading.py` | 更新已被 T18B 新入口替代的旧“不存在 build_and_load”断言，Attention 用户接口边界不变 |
| `flashinfer/data/api_parity.json` | 记录新接口与 Triton 实际功能程度，Ascend C 保持 framework |
| `tools/update_parity_inventory.py` | 使新接口和状态账本可重生成 |
| `.gitignore` | 忽略本地缓存根、Triton 缓存和 npubin |
| `docs/jit.md` | 声明 / 执行边界、真实格式、缓存实现、调用方式与限制 |
| `docs/reports/T18B-report.md` | 本次实测证据和验收结果 |

## 验证命令与实际输出

所有 Python 命令均使用用户指定的解释器。Triton 的三条 FutureWarning 未被过滤；
以下数值输出摘录省略重复的 warning 文本。

基线：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/ -q
```

```text
3 failed, 1432 passed, 4 warnings in 10.44s
```

最终代码在物理 NPU 1 串行回归：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 /root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/ -q
```

```text
FAILED tests/test_attention_metadata_wire.py::AttentionMetadataWireTests::test_config_directory_layout_and_canonical_blob_are_frozen
FAILED tests/test_attention_quant_launch_abi.py::AttentionQuantizedKVLaunchABITests::test_v1_is_frozen_and_v2_has_an_independent_binary_identity
FAILED tests/test_runtime_abi.py::KernelBinaryABITests::test_attention_binary_abi_matches_logical_order_mutation_and_stream
3 failed, 1453 passed, 4 warnings in 12.35s
```

三个失败名称与修改前基线完全相同，增加 21 个通过用例。一次在物理 NPU 0 同时运行
全量测试和缓存示例时，两边持续等待，已主动终止，未计为通过。该次设备 0 利用率
持续 100%，存在额外活跃设备进程；未确定等待根因。最终改用当时空闲的物理 NPU 1，
先完成全量回归，再完成缓存装载和两进程真实编译验证，没有对设备做全局 reset。

真实完整链路、相同 index 损坏重建及 CPU 数值参考：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b-final --rebuild-check
```

```text
disk_cache_hit: False
declared: triton/t18b/gemma 8b7180bceb4ebf00d5f9a4e760dc023b138352851e5038832b2c928777acdbb2
PASS corruption rebuild with the same immutable JitCacheIndex
PASS gemma_fused_add_rmsnorm float32 shape=(4,128) max_abs_error= 9.5367431640625e-07
PASS residual exact_equal= True
status: {"artifact_path": "/tmp/flashinfer-t18b-final/jit/.objects/8b7180bceb4ebf00d5f9a4e760dc023b138352851e5038832b2c928777acdbb2-459c84d180d343b18e83eec79fdbd2c8/gemma/kernel.bin", "entry_point": "_gemma_fused_add_rmsnorm_kernel", "is_compiled": true, "state": "artifact_recorded"}
```

新进程磁盘命中，禁止所有 subprocess（包括任何编译器探测）：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 /root/miniforge3/envs/flashinfer-a5/bin/python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b-final --cache-only
```

```text
disk_cache_hit: True
PASS gemma_fused_add_rmsnorm float32 shape=(4,128) max_abs_error= 9.5367431640625e-07
PASS residual exact_equal= True
PASS fresh-process cache load/launch with all subprocesses forbidden
```

真实 A5 两进程同时请求同一个新缓存中的 spec。用 `subprocess.Popen` 同时启动以下
命令两次（复验需换成尚未使用的 cache-dir）：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 /root/miniforge3/envs/flashinfer-a5/bin/python examples/jit_a5.py --cache-dir /tmp/flashinfer-t18b-final-concurrency
```

实际输出摘录：

```text
process 0 returncode 0
disk_cache_hit: False
PASS gemma_fused_add_rmsnorm float32 shape=(4,128) max_abs_error= 9.5367431640625e-07
PASS residual exact_equal= True
process 1 returncode 0
disk_cache_hit: False
PASS gemma_fused_add_rmsnorm float32 shape=(4,128) max_abs_error= 9.5367431640625e-07
PASS residual exact_equal= True
published_generations: 1
compiler_worker_logs: 1
PASS A5 two processes: one real compilation, both numerical checks passed
```

两边的状态查询均指向同一实际文件：

```text
/tmp/flashinfer-t18b-final-concurrency/jit/.objects/8b7180bceb4ebf00d5f9a4e760dc023b138352851e5038832b2c928777acdbb2-ea91646ec7074f20acf90c2a21536023/gemma/kernel.bin
```

入口都是 `_gemma_fused_add_rmsnorm_kernel`，`is_compiled=true`。

进程 / 线程并发安全、损坏缓存、原子发布、诊断错误与 18A 语义：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/test_jit_build.py tests/test_jit_trigger_layer.py -q
```

```text
87 passed in 3.22s
```

无 Triton / torch 可见、PATH 为空并禁止工具链探测的独立 `python -S` 进程：

```bash
/root/miniforge3/envs/flashinfer-a5/bin/python -m pytest tests/test_jit_trigger_layer.py::test_import_selection_and_declarations_with_no_optional_dependencies_or_compiler -q -s
```

```text
{"records": 79, "optional_import_attempts": [], "compiler_probe_attempts": [], "compiler_paths": {"bisheng": null, "ccec": null, "ascendc": null}, "selected_gemma": "triton/norm/gemma_fused_add_rmsnorm_v1"}
1 passed in 0.14s
```

修改前保存 `/tmp/flashinfer-t18b-selection-baseline.json` 中全部 79 个请求、选中 ID
和 descriptor 指纹。修改后逐条重放，输出：

```text
PASS 79/79 auto selections unchanged; 79/79 descriptor fingerprints unchanged
```

`python -m flashinfer parity-check`：`Parity ledger integrity: passed`。

## 尚未完成 / 需要确认的事项

Ascend C host wrapper、launch ABI 和真实设备装载尚未集成；原有三个 Attention / ABI
基线失败保留，未越界修复。旧 Attention 配方仍是声明，不自动变成真实 kernel。
本次不做跨模块 device linking、自动外部源码依赖追踪、缓存淘汰 / 旧代回收、
跨 Triton 版本运行时 ABI 适配。支持单个可信独立 Python kernel 模块和实测的
triton-ascend 3.6 ABI；编译目标与 torch / torch_npu / Python ABI 在装载时核对。
`DiskJitBuildStore(namespace=...)` 支持用户产物隔离，调用者须显式指定 namespace。
文件锁 / 原子指针替换要求本地 POSIX 文件系统。

没有阻塞已完成 Triton 路径的人为确认项。
