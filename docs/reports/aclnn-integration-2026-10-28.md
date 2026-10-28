# Torch / ACLNN 接入与回归验收（2026-10-28）

## 基线与实际改动

基线为 `infer` 的 `117db20e74c4c15481c3490186e0a73590845545`。此前的目录归一化已提交；本次在其上实现运行时代码，保留前一阶段文档核查内容，不重写历史报告。验证跨越 10 月 27 日晚至 28 日凌晨。

参考本地 vLLM-Ascend `96a69d793457917d5d8fdb6b3b4e18f6f372297f` 的 `csrc/aclnn_torch_adapter/op_api_common.h`、`NPUBridge.cpp`、`csrc/torch_binding.cpp` 及根 CMake。实际实现 Tensor → aclTensor、GetWorkspaceSize、workspace、当前 stream 和 `OpCommand`，没有只停留在接入说明。

| 位置 | 改动 |
| --- | --- |
| `csrc/aclnn_torch_adapter/adapter.cpp:87` | TorchNPU C++ 适配；storage 基址/offset 转换、格式/设备/重叠检查、两段式 API、workspace 与同步 eager 生命周期 |
| `csrc/aclnn_torch_adapter/adapter.cpp:171` | 通过 `OpCommand::RunOpApi` 提交，保留 Tensor/描述符/workspace 到设备完成；错误路径同样同步和释放 |
| `flashinfer/aclnn/_ops.py:74` | 两条生产算子的 Torch functional/out schema、PrivateUse1 与 FakeTensor 元数据实现，复用公共 facade 和原 dispatch 门禁 |
| `flashinfer/aclnn/_native.py:25`、`:32`、`:64` | 显式启用、版本与库 SHA-256 校验、延迟加载、只绑定已验证 op-package 的 API 地址 |
| `flashinfer/jit/tensor_adapter.py:103` | 增加可选原生适配分支；保留默认 Python 路径及原 workspace 观测边界 |
| `flashinfer/aclnn/build.py:13`、`setup.py:86` | 编译纯 Torch C++ host DSO，普通 wheel / CONFIG 打包；切换模式时清除旧适配产物 |
| `tests/test_aclnn_adapter.py:1`、`tests/test_aclnn_adapter_real.py:1` | 新增 5 个主机测试、2 个真机测试；已有 wheel 测试只增加产物与清理断言 |
| `README-wheel-conda.md`、`docs/aclnn-torch-adapter.md` | 打包、注册与直接调用方法，编译依赖及兼容边界 |

原有 `flashinfer.matmul_bias_silu`、`single_prefill_with_kv_cache_catlass` 的默认执行路径保持。完整 wheel 随包提供适配库，调用 `flashinfer.aclnn.enable_aclnn()` 后，两条原公共 API 与新 `torch.ops.flashinfer.*` 入口都走 C++/OpCommand → ACLNN。导入和注册不编译、不提前加载本仓适配库及算子包。

未改 recipe、冻结 spec、设备/host kernel 源码、ABI 常量、registry 精确集合或 `graph_capability`。原 CANN、Triton、plan/run、缓存、拒绝策略和图资源契约由完整原回归覆盖；额外验证启用 ACLNN 后同进程执行原 CANN RMSNorm。`compatibility-audit.json` 对 26 份关键冻结/构建文件核对为与基线字节相同。

## 环境与设备

在用户的 `dy_test` 环境验证，加载用户指定的 `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh`。原服务器 AGENTS 路径没有直接沿用。

```text
python: ../../tools/conda-envs/dy_test/bin/python3
Python 3.12.16
torch 2.10.0+cpu / torch_npu 2.10.0.post7
C++11 ABI: True
torch.npu.is_available(), torch.npu.device_count(): True 8
device: Ascend950PR_9579
CANN: ../../tools/miniconda3/Ascend/cann-9.2.0
CATLASS: ../catlass-ref @ 41bf90da655bba3c66d0acd7e00abe33960ecfd6
toolchain: 226cecda7bb975ee0ba3e39f449aac371f1ebc7cdcba90d6198678804884d250
```

原始版本、设备与参考文件摘要见 `environment.json`，该检查退出码 0。没有把原服务器结果当作本机结果。

## 实际执行与结果

证据目录相对于仓库为 `../../var/flashinfer/aclnn-integration-20261028-dlR5ZuJA/`，下述日志名相对于该目录。各命令先完整重定向，再记录真实退出码；没有用管道末端退出码冒充测试退出码。

激活上述环境，设置 `FLASHINFER_CATLASS_ROOT` 和 `FLASHINFER_TEST_PYTHON` 后，实际串行执行：

```bash
python3 -B -m flashinfer.aclnn.build --output-dir flashinfer/data/aclnn
python3 -B -m pytest -p no:cacheprovider -q --junitxml="$FI_ACLNN_WORK/full.xml"
python3 -B -m pytest -p no:cacheprovider -q -m real_compile --junitxml="$FI_ACLNN_WORK/real.xml"
```

| 验证 | passed | failed | errors | skipped | deselected | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 最终全量 | 1985 | 0 | 0 | 0 | 0 | 0 |
| 最终 real_compile | 82 | 0 | 0 | 0 | 1903 | 0 |
| 新增主机测试单跑 | 5 | 0 | 0 | 0 | 0 | 0 |
| 新增 ACLNN 真机测试单跑 | 2 | 0 | 0 | 0 | 0 | 0 |
| 原有 Torch 注册 / wheel 相关测试单跑 | 70 | 0 | 0 | 0 | 0 | 0 |

最终全量为 297.47 秒，4 条既有 warning；real_compile 为 303.50 秒。日志为 `full.log`、`real.log`，对应 `.xml` 与 `.exitcode` 同名保存。完整脚本为 `run-regression.sh`。没有放宽既有断言，没有把 SKIP 计入成功。

额外直接执行 `native-cold-jit.py matmul` 与 `attention`，各使用新的缓存目录、没有 AOT、JIT 允许。两条均为 `source=jit`、`compiler_calls=1`，经 C++/OpCommand 路径完成 functional/out 和原 API 数值调用，退出码均为 0；见 `native-cold-*.log`。这补验了原生适配下的缺失补建，没有只复用已有 wheel。

## 默认 wheel 与安装后调用

`build-wheel.sh` 清除 ALL/CONFIG/PREBUILT 等选择环境变量，以全新缓存实际执行普通构建：

```bash
python3 -B setup.py bdist_wheel -d "$FI_ACLNN_WORK/wheel"
```

构建退出码 0，两条生产算子均 `disk_cache_hit=false`，汇总 `requested=2, built=2, failed=0, skipped=0`。产物：

```text
wheel/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
bytes: 1630237
SHA-256: 44bfef64121cd039e10469b75c51e74bc58b632cdd5c99043c80809af5c2412f
```

`wheel-audit.json` 核对两个 AOT manifest、`libflashinfer_aclnn.so` 与适配库 manifest 均在包内；适配库摘要和源码摘要匹配，包内没有构建日志。`nm -D` 未发现未定义 CPython 符号，适配库仍依赖 Torch/TorchNPU C++ ABI，不能因 wheel tag 为 `py3-none` 就推断可跨 Torch 版本加载。实际编译命令保存在 `native-wheel-compile.log`。

`install-check.sh` 新建不含 `--system-site-packages` 的干净 venv，先 `--no-deps --force-reinstall` 安装本次 wheel，再以纯路径 `_deps.pth` 引入已有 Torch 等依赖。执行目录在仓库外，断言 `flashinfer.__file__` 位于该 venv；未将源码树调用当成 wheel 调用。

| 装后验证 | 实际结果 | 退出码 |
| --- | --- | --- |
| `python -B -m flashinfer selftest` | 2 PASS、0 FAIL、0 SKIP | 0 |
| Torch MatmulBiasSiLU functional/out + 原 API | C++/OpCommand，随包 AOT，compiler_calls=0 | 0 |
| Torch FlashAttention functional/out + 原 API | C++/OpCommand，随包 AOT，compiler_calls=0 | 0 |
| 启用 ACLNN 后的原 CANN RMSNorm | 两个独立算子进程中均数值通过 | 0 |
| 原 `flash_attention_acceptance.py` 无参数 | PASS flash_attention | 0 |
| 原 `matmul_bias_silu_acceptance.py` 无参数 | PASS matmul_bias_silu | 0 |

新 Torch 验证均使用非默认 stream、非零 storage offset，MatmulBiasSiLU 最大绝对误差为 `0.001665398273718699`，FlashAttention 为 `5.7678710803032995e-05`，容差分别保持 `atol=rtol=0.008` 和 `0.001`。原无参数脚本误差分别为 `0.003581114843043798` 和 `6.122825770479845e-05`。这些是数值结果，没有做性能比较。

默认 selftest 实际汇总：

```text
PASS gemm.matmul_bias_silu: bundled AOT hit; compilation=disabled
PASS ascendc.single_prefill_catlass: bundled AOT hit; compilation=disabled
Summary: 2 PASS, 0 FAIL, 0 SKIP (total=2)
```

详细输出、安装包路径和每项退出码见 `wheel-*.log` / `.exitcode`。完整的两条 PASS 原始行还含 CPU 参考误差与 spec；此处仅节录摘要。

## 源码模式与 PREBUILT 兼容

实际执行源码 wheel、sdist 以及 PREBUILT wheel 构建：

```bash
FLASHINFER_AOT_ALL=0 python3 -B setup.py -q bdist_wheel -d "$FI_ACLNN_WORK/source-only-wheel"
FLASHINFER_AOT_ALL=0 python3 -B setup.py -q sdist -d "$FI_ACLNN_WORK/sdist"
FLASHINFER_AOT_PREBUILT="$FI_EXISTING_RELEASE" \
  python3 -B setup.py -q bdist_wheel -d "$FI_ACLNN_WORK/prebuilt-wheel"
```

三条退出码均为 0。源码 wheel 无 AOT/原生适配库，带适配器源码；sdist 含适配器源码，没有生成的 `.so` 或日志。PREBUILT 在复用默认构建目录后携带两份 AOT、没有残留原生适配库。两种 wheel 构建的 `strace execve` 都没有启动 c++、bisheng、cmake 等编译进程，见 `source-mode-audit.json` 与 `prebuilt-audit.json`。

PREBUILT 另建隔离 venv，默认 selftest 仍为 2 PASS、0 FAIL、0 SKIP、退出码 0。显式 `enable_aclnn()` 后调用 Torch 算子，因为没有随包适配库而明确报 `ACLNN Torch adapter is not bundled`，检查命令退出码 0 表示**预期拒绝成立**，不表示缺失的原生能力运行成功。见 `prebuilt-install-check.sh`、`prebuilt-selftest.log`、`prebuilt-missing-native.log`。

## 发现并修正的兼容问题

首轮实现按“目录存在即自动启用原生桥”选择路径。实际全量为 **1979 passed、6 failed**、退出码 1；六项在访问 `workspace_requests[-1]` 时失败，原因是 C++ 直接调用 ACLNN 后，原测试对 Python `OpPackageModule.get_workspace_size` 的观测为空。日志和 JUnit 保留为 `auto-native-initial-full.*`。

最终改为**显式启用**，保留默认 Python 执行和观测边界，并新增默认路径不自动切换的断言。没有删除或修改 workspace 断言，没有削弱数值、动态 workspace 或冷 JIT 的验收。完整回归重新执行后得到上面的最终结果；不把首轮失败隐去，也不以原生单例通过代替旧特性回归。

新增主机测试首次单跑为 3 passed、2 failed，原因是 Torch 的 backend 自动导入与测试中的 TorchNPU 隔离桩冲突。仅在主机隔离子进程设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，随后 5 passed；真机运行没有关闭 backend。初始结果保留为 `adapter-unit-initial.*`。

本次推翻了“新增接口只需说明现有 Python ACLNN 调用”的交付结论：已实际加入原生 host 适配库及 Torch schema。没有改变此前对动态图、任意参数 ABI 或性能未验证的结论。

## 文档核查与未验证事项

对比图已增补可选 Torch 入口与 C++/OpCommand 边界，Mermaid 11.4.1 解析为 `flowchart-v2`，退出码 0。`framework_design.md` 保留 CRLF，围栏数量从基线 60 增至 62；新增算子指南保持 8，wheel README 从 18 增至 20。未搬动既有章节，未减少代码块。

- 两条自研生产算子的 capture/replay、异步返回、自动微分和完整模型接入：未验证；`graph_capability` 未提升。
- 通用 scalar/attribute、optional Tensor/Tensor list、数据内容驱动 tiling：未实现或未验证；当前桥仅支持完整档的有序 Tensor ABI。
- 多 package identity 同进程：原限制保留，两个生产包仍分别启动进程；未宣称解除。
- 其他 SoC、Torch/TorchNPU/CANN 版本与完全独立下载依赖的新 conda 环境：未验证；本次隔离 venv 复用已安装依赖的纯路径。
- vLLM-Ascend 自身的编译/运行及性能优势：未验证，未做性能测量。
- 原服务器 `-9/SIGKILL` 的信号发送者：未定位。本机本轮冷编译通过，不据此认定原因为守护进程、OOM 或源码错误。

原始日志、退出码、JUnit、构建脚本、wheel 与变更快照保存于上述证据目录；归档为同目录名后缀 `-evidence.tar.gz`，校验文件为 `.sha256`。虚拟环境和生成的缓存树不进入归档。
