# 目标机器编译与回归验证（2026-10-27）

## 验证基线与边界

- 分支为 `infer`，开工 HEAD 为 `3ece69a`，开工工作区干净。
- `git fetch origin` 成功后仍找不到 `5060a5c`；交接文件没有第 0 节，所指定的两份后续报告也未包含在此 checkout 中。用户确认改以当前 checkout 正常验证。
- 本报告记录目标机器的实际执行结果，不把原服务器的回归数字作为当前机器的通过证据。既有设计文档与历史报告未重写。
- 完整日志、退出码、构建产物和重冻候选补丁保存于 `build/regression-20261027-target/`。该目录受 Git 忽略；迁移时需要单独复制。
- 测试与构建日志、JUnit 数值记录、补丁和新 wheel 另归档为 `build/regression-20261027-target-evidence.tar.gz`，文件清单与摘要随归档保存。

## 环境与设备自检

原服务器解释器无法执行，shell 返回 `Permission denied`、退出码 `126`。按用户指定的环境脚本建立了独立 Python 3.11 测试环境，没有修改共享 Python 环境。

| 项目 | 当前实测 |
| --- | --- |
| 环境脚本 | `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh` |
| 测试解释器 | `../../tools/conda-envs/flashinfer-a5/bin/python`，Python 3.11.17 |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post7` |
| Triton Ascend / pytest / NumPy | `3.6.0` / `8.4.2` / `1.26.4` |
| CANN | `9.2.0`，安装于 `../../tools/miniconda3/Ascend/cann-9.2.0`；不是原服务器的 beta.2 |
| bisheng | clang 15.0.5，构建时间 `2026-10-15T17:21:21+08:00` |
| catlass | `../catlass-ref`，固定 revision `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| golden generator SHA-256 | `c988b56dc2113990f2d0978185db3eda2cd18db05018b2f042ae3c2f4a6f2fb9`，与测试固定值相同 |
| 设备自检 | `True 8`；设备名称 `Ascend950PR_9579` |

实际执行：

```bash
source build/regression-20261027-target/env.sh
"$FLASHINFER_TEST_PYTHON" -B -c \
  "import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())"
```

同等检查的完整版本输出保存在 `torch-device-check.log`，退出码 `0`。另独立调用 `aclrtGetDeviceCount` 返回 `0`、设备数 `8`。`npu-smi` 原始状态保存在 `environment-inventory.log`，其中部分卡显示 Alarm；不将设备可见性等同于所有卡健康或全部经过数值验收。

Triton Ascend 3.6.0 使用本机保留的 wheel，SHA-256 为 `8775838a0419a6f4ce191c28493dbeb3829081db6ec973ee7c62b20646df0a54`；其原始来源未验证。Triton 当前 target 实测为 `GPUTarget(backend='npu', arch='Ascend950PR_9579', warp_size=0)`。golden 生成器额外依赖的 `ml_dtypes==0.5.4` 已安装；没有通过替换 NumPy 1.26.4 解决依赖。

## 最小编译与生产构建

| 实际执行项 | 退出码 / 结果 | 日志或证据 |
| --- | --- | --- |
| 普通 host `g++ -O2 -c` | `0` | `probe/host.log`、`probe/results.json` |
| 无依赖 AIC-only 空 kernel | `0`，ELF machine `0x1029` | `probe/aic_empty.log`、`probe/empty.o` |
| AIV 写值 kernel | `0` | `probe/aiv_store.log` |
| 包含 `kernel_operator.h` 的 AIC-only 空 kernel | `0` | `probe/cann_empty.log`、`probe/cann_empty-result.json` |
| native matmul_probe op-package | `0`，冷构建 | `native-aot.log`、`native-cache/` |
| catlass matmul_probe op-package | `0`，冷构建 | `catlass-probe-aot.log`、`catlass-probe-cache/` |
| 重冻前生产 AOT | `1`，工具链指纹不匹配，编译器启动前拒绝 | `production-aot-before.log` |
| 授权重冻后生产 AOT | `0`，`requested=2, built=2, failed=0, skipped=0`，两条均为冷构建 | `production-aot.log`、`production-cache/` |
| 携带两条产物的 wheel 构建 | `0` | `wheel-build.log`、`wheelhouse/` |
| 干净 venv 强制安装 wheel | `0`，已断言 import 来自该 venv | `wheel-install.log`、`wheel-identity.json` |
| 在源码树外运行新 wheel 的 selftest | `0`，`2 PASS, 0 FAIL, 0 SKIP` | `selftest.log` |

最小 CANN 编译使用以下实测参数；完整 argv 与真实进程 returncode 保存在对应 JSON 中：

```bash
"$ASCEND_HOME_PATH/bin/bisheng" -xcce \
  --cce-aicore-arch=dav-c310-cube --cce-aicore-only -std=c++17 -O2 \
  -I"$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw" \
  -I"$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface" \
  -I"$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl" \
  -c build/regression-20261027-target/probe/cann_empty.cpp \
  -o build/regression-20261027-target/probe/cann_empty.o
```

生产构建实际使用现有 `cann930` 环境中的 Python 执行编译驱动。已分别实测该解释器和新测试解释器的工具链指纹均为 `90c5a62ba29c10d626b9da96823b1630c07936d67a16652230f4d43b58f43637`，见 `frozen-environment.log`；没有加载其 CANN 9.3 环境。

```bash
source build/regression-20261027-target/env.sh
export PATH=/opt/miniconda3/envs/cann930/bin:$PATH
/opt/miniconda3/envs/cann930/bin/python -B -m flashinfer.aot --all \
  --cache-dir build/regression-20261027-target/production-cache \
  --output-dir build/regression-20261027-target/production-release \
  > build/regression-20261027-target/production-aot.log 2>&1
result=$?
printf '%s\n' "$result" > build/regression-20261027-target/production-aot.exitcode
```

输出的两个新 spec 为 `5d51d67544573c17abe93c70372d41a6d0038dd522aa58b9d7d4e638f879f2ce`（MatmulBiasSiLU）和 `3a96bbfe762dd3b72e33f1142cdce0e569e75e941484e5625a7b55901eb6b995`（FlashAttention）。

当前机器没有复现原服务器的 `-9/SIGKILL`。本次定位到的生产构建阻塞是迁移后的工具链身份不匹配；这与旧机器的 SIGKILL 是不同现象。旧故障的信号发送者、是否涉及守护进程或 OOM，均仍未验证。

## 实际改动

- `flashinfer/data/kernels/ascendc_specs.json:129`：经用户授权更新两条 spec 的 `name`、`environment_fingerprint` 和 `resolved_recipe`，共替换 6 行。两条 recipe、ABI、源码摘要、descriptor 指纹、特化范围及登记顺序不变；`registry.json` 字节未变。
- `tests/flash_attention_acceptance.py:20`、`tests/test_flash_attention_real.py:55`：允许配置迁移后的解释器与 catlass 路径，并通过 `FLASHINFER_TEST_CATLASS_ROOT` 在子进程中保留 golden checkout 路径。隐藏构建依赖时仍使用原固定 oracle；解释器一致性、8 卡、固定 revision、golden 摘要及数值断言均保留。
- `tests/test_aot_loader_paths.py:36`、`tests/test_ascendc_op_package.py:139`：先用 monkeypatch 登记环境变量，再删除它。原先在变量不存在时调用 `delenv(..., raising=False)` 不会登记撤销操作，loader 后续直接写入的伪造 vendor 路径会泄漏给后面的真实用例。
- 新增本报告；既有 Markdown 文件没有改动，没有搬动章节或减少代码块。

4 个修改过的测试/验收脚本共 100 条 `assert` 的 AST 与 HEAD 完全相同，记录于 `assertion-check.json`。登记表精确集合、设备检查、数值容差和编译禁用护栏没有放宽。

## 回归与装后验证

首轮串行执行结果完整保存在 `initial/`：全量 `1966 passed, 12 failed, 0 errors, 0 skipped`，退出码 `1`；`real_compile` 为 `74 passed, 6 failed, 0 errors, 0 skipped, 1898 deselected`，退出码 `1`。

失败分为三类：

1. 5 个 FlashAttention 场景在导入固定 golden 生成器时因缺少 `ml_dtypes` 显式失败。
2. 1 个 FlashAttention 禁编译场景中，初版迁移配置把 golden checkout 路径也改为刻意不存在的构建依赖路径。已用独立的测试路径配置修复，未改 oracle 或断言。
3. 全量独有的 6 个真实执行失败返回 `GetWorkspaceSize=561002`。CANN 日志记录 `The tiling function does not exist`。相关单元 fixture 向后续进程泄漏伪造的 `ASCEND_CUSTOM_OPP_PATH`，污染了真实 vendor 发现；独立 real_compile 没有这些单元 fixture，相关 6 例均通过。

环境泄漏通过进程内 `pytest.main` 独立复现：修复前单元测试显示 `1 passed`，但测试后变量由未设置变为伪造 vendor 路径，外层断言失败、退出码 `1`；修复后两个相关文件 `34 passed`，测试前后变量均未设置，外层退出码 `0`。见 `fixture-env-before.log`、`fixture-env-after.log`。这也说明不能仅凭 `561002` 就认定环境根路径被 unset；本轮没有 unset CANN 根路径。

修复后的定向验证为 `37 passed, 0 failed, 0 skipped`，退出码 `0`；该次先运行 34 个伪造运行时的单元用例，再运行 native AOT、FlashAttention 禁编译 AOT、MatmulBiasSiLU 8 组数值验收，验证顺序污染已消除。完整日志为 `targeted.log`。

最终结果如下；JUnit 时间戳确认两套回归串行执行，汇总为 `summary.json`。

| 验证 | passed | failed | errors | skipped | deselected | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 全量 | 1978 | 0 | 0 | 0 | 0 | 0 |
| real_compile | 80 | 0 | 0 | 0 | 1898 | 0 |
| 定向验证 | 37 | 0 | 0 | 0 | 0 | 0 |
| 新 wheel selftest | 2 | 0 | — | 0 | — | 0 |

实际执行入口为 `bash build/regression-20261027-target/run-regressions.sh`，脚本先后执行以下两条命令，并分别在 `tail` 之前保存 `$?`。JUnit 记录包含通过用例的 stdout，便于直接查看数值和编译计数。

```bash
source build/regression-20261027-target/env.sh
"$FLASHINFER_TEST_PYTHON" -B -m pytest -p no:cacheprovider -q \
  -o junit_logging=all -o junit_log_passing_tests=true \
  --basetemp=build/regression-20261027-target/full-final-temp \
  --junitxml=build/regression-20261027-target/full.xml \
  > build/regression-20261027-target/full.log 2>&1
full_result=$?
printf '%s\n' "$full_result" > build/regression-20261027-target/full.exitcode

"$FLASHINFER_TEST_PYTHON" -B -m pytest -p no:cacheprovider -q \
  -o junit_logging=all -o junit_log_passing_tests=true -m real_compile \
  --basetemp=build/regression-20261027-target/real-final-temp \
  --junitxml=build/regression-20261027-target/real.xml \
  > build/regression-20261027-target/real.log 2>&1
real_result=$?
printf '%s\n' "$real_result" > build/regression-20261027-target/real.exitcode
```

实际输出：

```text
1978 passed, 4 warnings in 307.71s (0:05:07)
FULL_EXIT_CODE=0
80 passed, 1898 deselected in 353.50s (0:05:53)
REAL_COMPILE_EXIT_CODE=0
Summary: 2 PASS, 0 FAIL, 0 SKIP (total=2)
```

全量的 4 条 warning 为 torch_npu 基础格式提示及 3 条 Triton 弃用提示，原文保留在日志中。两套回归都直接记录了 native probe、FlashAttention、MatmulBiasSiLU 的 `source=jit, compiler_calls=1, result=PASS`；两条生产算子的 `toolchain_calls` 也均为 `1`。提取记录分别保存在 `full-numerical-evidence.json`、`real-numerical-evidence.json`。这不是只编译不执行的验证。

新 wheel 的装后自检在干净 venv、源码树外执行，未使用 `--system-site-packages`；强制安装后才添加纯路径 `_deps.pth`，且已断言 `flashinfer.__file__` 位于该 venv。JIT 禁用，两条均命中包内 AOT：MatmulBiasSiLU 的最大绝对误差 `0.00358111`，FlashAttention 为 `6.12283e-05`，均通过各自既有容差。此结果仅覆盖 selftest 样本。

实际 wheel 为 `wheelhouse/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`（位于上述日志目录），SHA-256 为 `fb697f605b5a5d34d79d2b7d47a2d5836ab5bf3022d0c2129b7e2d764e5e254b`。ZIP 完整性及其中 177 个源码/构建输入文件与最终工作区的一致性均已核对，见 `wheel-source-check.json`。

## 仍未验证

- 原服务器 `-9/SIGKILL` 的根因及信号发送者。
- 缺失提交 `5060a5c` 所代表的代码与文档状态。
- 原服务器 CANN beta.2 产物与当前 CANN 9.2.0 的二进制兼容性；本次使用当前工具链重新构建的新产物。
- 多 SoC、全部 8 张卡逐卡数值、无编译器容器镜像及下游完整模型集成。
- 同口径性能测量；本报告不作性能优势判断。
