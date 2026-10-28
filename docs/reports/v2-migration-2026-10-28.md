# Ascend C 对外统一 v2 迁移验收（2026-10-28）

## 范围与实际改动

按用户确认的 [归一评审](recipe-unification-review-2026-10-28.md) 实施：**对外只提供 v2 新建、选择、解析构建和发布入口**。旧 v1 artifact 仍按原身份内部读取，旧裸指针编译/调用覆盖保留在测试设施，不再作为公开源码或配置选项交付。

分支为 `infer`，HEAD 为 `117db20e74c4c15481c3490186e0a73590845545`；迁移基于此前已经完成、尚未提交的 Torch/ACLNN 工作区。开工快照保存在本轮证据目录 `before/`，没有覆盖上一轮改动。

| 位置 | 最终行为 |
| --- | --- |
| `flashinfer/jit/recipe.py:326`、`:394`、`:400`、`:525` | 公开 `BuildRecipe` 默认 schema=2，拒绝 v1；配置只接受 op_package，选择/构建解析拒绝旧 recipe；原校验与序列化保留在私有基础实现 |
| `flashinfer/jit/_legacy_recipe.py` | 旧测试与兼容设施显式使用私有 reader/resolver/export helper；不自动升级旧 metadata |
| `flashinfer/aot/build.py:43` | 新 release/PREBUILT 导出拒绝 v1，且拒绝时不创建目标目录；探针声明绑定实际 tiling schema |
| `flashinfer/jit/op_package.py` | 读取已有 artifact 时区分新声明与旧格式，原 inventory、ELF、参数、目标和同步门禁保持 |
| `flashinfer/jit/torch_library.py:48` | 公开探针只接受 v2 + 已验证 spec，复用通用 `TorchOpPackageRunner`；旧桥移到私有模块 |
| `csrc/recipes/matmul_probe.json` | 五条 op-package 探针全部迁移到 v2；公开配方合计七条，均为 v2 |
| `csrc/op_package/matmul_probe.cpp:18`、`csrc/include/flashinfer_ascend/probe_tiling.h:5` | 命名 host callback，显式 version=2/size=256 的八字节 tiling；device 同时核验版本和大小 |
| `tests/fixtures/legacy_ascendc/` | 保留七条原 v1 配方、原源码闭包与两条 host_shared 探针；原 source digest、序列化和指纹断言不变 |
| `csrc/CMakeLists.txt`、`examples/ascendc*.json`、JIT/Torch 示例 | 公开构建与示例只走 v2 ACLNN；裸指针头、launcher 与 HostShared CMake 移出公开 csrc |
| `tests/test_ascendc_v2_public.py` | 新增五项主机测试和两项真机测试，精确钉死七条公开声明、拒绝边界、旧身份和通用 Tensor runner |

探针仍为 FP32 固定 16×16×16、三 Tensor、AIC-only、零 workspace 声明；以统一 host callback 返回限定支持域的常量 tiling，不扩大 shape 或数学语义。没有新增生产 descriptor，登记表仍为 81 条 / 43 个 op，两条自研生产条目不变。

`ascendc.op_package.v1`、SourceRef/JitSpec/原生桥 manifest 的格式版本仍各自保留；这些数字不是对外算子接入档位。内部 v1 reader 的保留不意味着还允许公开构建或发布 v1。

## 生产 spec 与兼容边界

公共 CMake 与 loader 文件内容属于构建身份闭包。按照已确认的迁移范围，使用两条生产发布工具的 `freeze_contract()` 重新冻结声明；没有关闭指纹校验，也没有让新声明匹配旧缓存。

```text
旧 toolchain: 226cecda7bb975ee0ba3e39f449aac371f1ebc7cdcba90d6198678804884d250
新 toolchain: e3ffa08572a756f7999952e35d2080285216d280f9f0a72c489c4b51ba66d62a
```

`refreeze-summary.json` 记录两个前后身份。两条生产 recipe、host/device 源码与 descriptor 没有变化；registry 内容和顺序保持原字节，仅冻结 spec 的环境/解析身份更新。生产 dtype、shape/envelope、workspace 上界和 `graph_capability` 未扩大。

升级源码后应重新构建本轮 wheel，不应把上一轮 wheel 或其 AOT spec 当作本轮产物。旧 artifact 不会被悄悄改写为 v2，旧格式的损坏/越界仍由原校验拒绝；旧配方不再被公开 publisher 接受。

## 环境与证据

证据目录为仓库相对路径 `../../var/flashinfer/v2-migration-20261028-vtv58qp4/`，以下日志名均相对于该目录。使用用户 `dy_test`，加载 `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh`；设备自检为 **True 8**。解释器为 Python 3.12，Torch/TorchNPU 与上一轮相同；本轮没有使用 AGENTS 中原服务器的解释器/SDK 路径。

每个阶段完整重定向 stdout/stderr 后立即保存真实退出码，日志不放在可被清理的 build/dist 中。`run-regression.sh` 串行运行全量和 real_compile，并保存 JUnit，包括成功用例的可捕获输出。

本轮有失败和主动中断记录，均保留，最终结果另列：

- 首次主机收集暴露迁移中的类型注解/重复关键字问题，修正实现后相关旧契约集 145 passed；新增公开边界集随后通过。
- 针对性真机首次为 11 passed、1 failed、40 deselected。失败是跨 SoC 拒绝场景的 Python 子进程超时 180 秒，首次没有 stdout/stderr。保持原超时和断言，同一产物直接复现返回 PASS、compiler_calls=0；未定位首次超时原因。
- 上述超时日志中的 `returncode=-9` 是 Python 测试子进程被超时清理，不是 bisheng 编译失败，不能据此认定守护进程、OOM 或源码错误。
- 一次完整回归在交付空白检查发现探针 EOF 空行后主动中断，记录为 `interrupted-full.*`；其 21 passed 不计入最终全量。清理空白后重新计算探针源码摘要。
- 首次完成的全量为 1991 passed、1 failed。旧路径测试用 v1 合成 fixture 调用新公开 publisher，触发预期的 v2 门禁。仅将该 fixture 生成切换到内部 helper；原注册路径、FD、异常和精确集合断言保持。记录为 `initial-full.*`。
- 修正 fixture 边界后，相关主机集为 140 passed、4 deselected、退出码 0；随后重新执行完整回归。
- 最终全量为 1992 passed，0 failed/errors/skipped，退出码 0。随后第一次独立 real_compile 为 83 passed、1 failed；失败发生在输入首次 `.npu()` 的 `LazySetDevice`，报 507033/E39007（设备子进程启动超时），未进入算子调用。诊断时进程阻塞于 `hdcdrv_recv_peek`；gdb 附加被主机 ptrace 策略拒绝，没有修改该策略。
- 设备重新分配/拷贝检查随即通过：`available True devices 8`、`DEVICE_ALLOCATION_COPY_PASS npu:0`，退出码 0。保留 `initial-real.*` 和诊断日志，原配置、原超时、原断言完整重跑 real_compile；没有重置共享设备或跳过失败项。

## 最终回归与打包结果

实际命令由 `run-regression.sh` 和原配置重跑的 `run-real.sh` 保存：

```bash
python3 -B -m pytest -p no:cacheprovider -q --junitxml="$FI_V2_WORK/full.xml" -o junit_logging=all
python3 -B -m pytest -p no:cacheprovider -q -m real_compile --junitxml="$FI_V2_WORK/real.xml" -o junit_logging=all
```

| 验证 | passed | failed | errors | skipped | deselected | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 最终全量 | 1992 | 0 | 0 | 0 | 0 | 0 |
| 最终 real_compile | 84 | 0 | 0 | 0 | 1908 | 0 |
| 迁移相关主机集 | 140 | 0 | 0 | 0 | 4 | 0 |

全量耗时 328.36 秒，4 条既有 warning；real_compile 耗时 321.08 秒。相对迁移前的 1985/82，新增五项公开契约主机测试和两项 v2 探针真机测试。两条旧 host_shared/op-package 的编译、FakeTensor、namespace、数值与正控制覆盖继续运行，没有删除旧断言来换通过。

完整原始输出、JUnit 与真实退出码为 `full.*`、`real.*`、`host-final.*`。首次失败、主动中断、硬件初始化失败和复现另存，未覆盖为成功日志。

### 探针目标矩阵

五条公开探针均已重新编译并通过标准包验证：

| 目标 / 实现 | 编译与包校验 | 设备数值 |
| --- | --- | --- |
| Ascend950 / native | PASS | PASS，含 C++/OpCommand、FakeTensor 零设备分配、namespace、opcheck 和正控制 |
| Ascend950 / CATLASS | PASS | PASS，同上 |
| ascend910b1 / native | PASS，实际来自 AOT 跨 SoC 候选拒绝测试 | 未验证，在 Ascend950 侧确认其目标不匹配被拒 |
| ascend910_9391 / native | PASS，独立冷交叉编译 | 未验证 |
| ascend310p1 / native | PASS，独立冷交叉编译 | 未验证 |

后两条实际执行：

```bash
python3 -B -m flashinfer.aot --config examples/ascendc_aot_native.json \
  --soc ascend910_9391 --soc ascend310p1 \
  --cache-dir "$FI_V2_WORK/probe-multi-soc-cache" \
  --output-dir "$FI_V2_WORK/probe-multi-soc-release"
```

退出码 0，两个输出均 `disk_cache_hit=false`，详见 `probe-multi-soc.log`；不将交叉编译记作其他芯片的设备通过。

### wheel、安装与源码模式

`build-wheel.sh` 清除 ALL/CONFIG/PREBUILT 选择变量、使用新缓存，执行普通 `python3 -B setup.py bdist_wheel`。两条生产算子均冷构建，退出码 0，汇总为 `requested=2, built=2, failed=0, skipped=0`。

```text
wheel/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
bytes: 1630809
SHA-256: 18f879cb29dc169028e35882af121ebfddfb82e86baf161e9ea3676e718c2967
```

`wheel-audit.json` 确认两份 AOT、C++ 适配库与 manifest、七条 v2 构建配方；无旧裸指针构建输入、旧测试源码或构建日志。适配库 SHA-256/源码摘要匹配，未发现未定义 CPython 符号；运行仍要求匹配的 Torch/TorchNPU C++ ABI。

`install-check.sh` 创建干净 venv，未使用 `--system-site-packages`；先强制安装本轮 wheel，再用纯路径 `_deps.pth` 引入已有依赖。从仓库外执行并断言 `flashinfer.__file__` 位于该 venv，不把源码导入当作装后验证。

| 装后验证 | 实际结果 | 退出码 |
| --- | --- | --- |
| `python -B -m flashinfer selftest` | 2 PASS、0 FAIL、0 SKIP | 0 |
| 新 Torch MatmulBiasSiLU functional/out + 原 API | C++/OpCommand、随包 AOT、compiler_calls=0，最大绝对误差 0.001665398273718699 | 0 |
| 新 Torch FlashAttention functional/out + 原 API | C++/OpCommand、随包 AOT、compiler_calls=0，最大绝对误差 5.7678710803032995e-05 | 0 |
| 启用 ACLNN 后的原 CANN RMSNorm | 两个独立进程均数值 PASS | 0 |
| 原 `flash_attention_acceptance.py` 无参数 | PASS flash_attention | 0 |
| 原 `matmul_bias_silu_acceptance.py` 无参数 | PASS matmul_bias_silu | 0 |

新 Torch 验证继续覆盖非默认 stream、非零 storage offset，容差保持 0.008（MatmulBiasSiLU）与 0.001（FA）。随包生产 spec 分别为 `bdda829d190f6d644420dd40a64e09b52e928771399672c1c8f0a9b737054bab` 与 `dba45e0318bbd240c5e1a9ed32687b50ce7a1aa0cd6353f29215eb1b4518417d`。

源码模式与 sdist 构建退出码均为 0，`source-delivery-audit.json` 确认公开配方只有七条 v2、旧 raw 构建输入和测试源码均不交付、内部兼容 reader 保留，源码 wheel 构建没有 compiler exec。PREBUILT 从本轮 wheel 中提取的已验证两份 AOT 打包，同样未执行编译器，未携带残留原生桥；另建隔离 venv 后 selftest 为 2 PASS、0 FAIL、0 SKIP。

PREBUILT 显式启用未随包提供的原生桥时，检查命令确认 `ACLNN Torch adapter is not bundled`、fallback=false。该命令退出码 0 表示预期拒绝成立，不表示缺失能力执行成功。见 `prebuilt-*.log`、`prebuilt-audit.json`。

### 文档与交付核查

`framework_design.md` 已同步公开仅 v2、内部旧 reader、探针迁移与重新冻结身份；新增算子指南、AOT 指南和编译选项说明同步更新。主设计文档保留 CRLF，围栏数量保持 62；新增算子指南保持 8，AOT 指南保持 28。Mermaid 解析为 `flowchart-v2`，退出码 0。

`migration-scope-audit.json` 验证 13 份生产文件/登记表与迁移前字节相同，16 份旧快照原字节相同；两份冻结 spec 的差异仅为 name、environment_fingerprint 和 input_fingerprints 中的 resolved_recipe。旧冻结断言未删除或放宽。

原始日志、退出码、JUnit、脚本、wheel 及变更快照归档为证据目录同名后缀 `-evidence.tar.gz`，同名 `.sha256` 保存校验值；缓存与依赖虚拟环境不进入归档。

## 仍未验证

- 其他 SoC 的设备数值与跨版本 Torch/TorchNPU/CANN 兼容；声明或交叉编译不等于设备验证。
- 两条生产自研算子的 aclgraph capture/replay、异步返回、自动微分、完整模型和性能；Torch compile/FakeTensor 检查不提升图能力。
- 任意 scalar/attribute、optional Tensor/Tensor list、数据内容驱动 tiling 的通用接入。
- 原服务器 SIGKILL 的发送者，以及本轮单次测试子进程超时的具体原因。
- 旧 v1 reader 的最终退役；本轮实现对外 v2，内部兼容没有删除。
