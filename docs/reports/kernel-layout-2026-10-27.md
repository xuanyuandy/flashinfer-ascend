# 设备源码目录归一化验证（2026-10-27）

## 基线与目录选择

基线为 `infer` 的 `f18c1de`，开工工作区干净。使用用户的 `dy_test` 环境与 CANN 环境脚本，设备自检实际输出 `True 8`，退出码 0。

本次直接查阅本机竞品 checkout，并核对引用文件与其 HEAD 字节一致，没有将远端最新版本作为本次依据：

| 参考仓库 | 本地 commit | 已核对位置 |
| --- | --- | --- |
| vLLM-Ascend | `96a69d793457917d5d8fdb6b3b4e18f6f372297f` | `../vllm-ascend/CMakeLists.txt:54` 从 `csrc/kernels/*.cpp` 收集直接设备内核，另有按算子组织的 `csrc/**/op_kernel/` |
| flashinfer-ori | `62b5ae4e52e14d841e1b0455dc3cfa7053181741` | flashinfer-ori: `csrc/`、`include/flashinfer/`；`flashinfer/jit/env.py:399`、`:400` 定义包内 `data/include`、`data/csrc` |

因此采用 `csrc/kernels/ascendc/` 作为本仓唯一 Ascend C 设备源码根，沿用已有生产算子的位置，按算子分目录。没有复制竞品的递归扫描或后端实现。

## 实际迁移与身份边界

| 原位置 | 现位置 |
| --- | --- |
| `kernels/ascendc/matmul_probe/native.cpp` | `csrc/kernels/ascendc/matmul_probe/native.cpp` |
| `kernels/ascendc/matmul_probe/catlass.cpp` | `csrc/kernels/ascendc/matmul_probe/catlass.cpp` |
| `kernels/manifest.schema.json` | `flashinfer/data/kernels/manifest.schema.json` |
| `kernels/README.md` | `csrc/kernels/README.md`，更新目录职责及过时的空登记表说明 |
| `kernels/ascendc/.gitkeep`、`kernels/triton/.gitkeep` | 删除占位文件，移除顶层 `kernels/` 空目录 |

两个探针源码与 schema 均逐字节迁移。`csrc/recipes/matmul_probe.json` 的 7 条变体只更新 `source`、`sources` 中的 14 个 locator；摘要、ABI、编译选项、依赖和约束不变。探针使用新路径形成新的构建身份，不能把旧路径的缓存命中当作迁移后的编译证据。

生产源码未迁移；以下文件与基线逐字节一致，无需重新冻结生产声明：

- `flashinfer/data/kernels/ascendc_specs.json`
- `flashinfer/data/kernels/registry.json`
- `csrc/recipes/matmul_bias_silu.json`、`csrc/recipes/flash_attention.json`
- `csrc/include/flashinfer_ascend/launch_abi.h`

关键代码位置：

- `setup.py:65`：构建输入仅从统一的 `csrc/` 收集。
- `setup.py:77`：复制前清理生成的 `flashinfer/data/build/`，避免复用 build 目录时携带旧 `data/build/kernels/` 文件。
- `MANIFEST.in:1`：sdist 由 `csrc/` 规则收集 host/device/CMake/recipe，不再单列顶层 kernel 树。
- `tests/release_checks/wheel_checks.py:56`：在原默认/显式 AOT 打包测试中预置旧路径残留，要求构建后旧树消失、新路径两份源码字节正确；原有 14 条 assert 保留，增加 2 条 assert 表达式。

`flashinfer/data/kernels/` 存放运行时声明与 schema，`csrc/op_package/kernel/` 存放 CANN 构建模板；设备实现均在 `csrc/kernels/ascendc/`。Triton Python 实现继续位于 `flashinfer/triton/`。

设计文档及构建指南做增量路径更新，历史报告和任务记录未改写。`docs/framework_design.md` 保持 CRLF，代码块围栏数量仍为 60；其余修改的既有指南代码块数量也保持不变。空白检查使用 `git -c core.whitespace=cr-at-eol diff --check`，识别该文件原有换行格式。

## 环境、命令与日志

本轮证据目录为 `../../var/flashinfer/kernel-layout-20261027-LG2E1QlS/`，位于仓库的 build/dist 之外。文中日志名均相对于该目录。

实际设备检查：

```bash
source ../../tools/miniconda3/etc/profile.d/conda.sh
conda activate dy_test
source ../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
python3 -B -c 'import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())'
```

实际输出 `True 8`，退出码 0，保存于 `device-check.log`。完整依赖列表为 `pip-freeze.txt`，开工源码摘要为 `baseline.json`，竞品引用记录为 `competitor-layout.json`，迁移、断言及文档代码块核对为 `layout-review.json`。

定向执行：

```bash
python3 -B -m pytest -p no:cacheprovider -q \
  tests/test_ascendc_recipe.py tests/test_ascendc_op_package.py tests/release_checks/wheel_checks.py
```

实际输出 `53 passed, 1 warning in 0.52s`，退出码 0，`0 skipped`。warning 为 wheel 命令导入路径的弃用提示，原文保留在 `targeted.log`。

## 串行回归

实际入口为证据目录中的 `run-regressions.sh`。设置 `FLASHINFER_TEST_PYTHON` 为 `dy_test` 当前解释器、`FLASHINFER_CATLASS_ROOT` 为本机固定 checkout，并清除禁编译和外部 AOT/vendor 搜索覆盖后，按顺序执行：

```bash
python3 -B -m pytest -p no:cacheprovider -q \
  -o junit_logging=all -o junit_log_passing_tests=true \
  --basetemp="$FI_LAYOUT_DIR/full-temp" --junitxml="$FI_LAYOUT_DIR/full.xml" \
  > "$FI_LAYOUT_DIR/full.log" 2>&1
FI_FULL_RC=$?
printf '%s\n' "$FI_FULL_RC" > "$FI_LAYOUT_DIR/full.exitcode"

python3 -B -m pytest -p no:cacheprovider -q -m real_compile \
  -o junit_logging=all -o junit_log_passing_tests=true \
  --basetemp="$FI_LAYOUT_DIR/real-temp" --junitxml="$FI_LAYOUT_DIR/real.xml" \
  > "$FI_LAYOUT_DIR/real.log" 2>&1
FI_REAL_RC=$?
printf '%s\n' "$FI_REAL_RC" > "$FI_LAYOUT_DIR/real.exitcode"
```

全量实际结果为 `1978 passed, 4 warnings in 415.59s (0:06:55)`，退出码 0。JUnit 确认失败、错误、跳过均为 0。warning 为 torch_npu 基础格式及三条 Triton 弃用提示，原文保留在日志中。

独立 `real_compile` 实际输出为 `80 passed, 1898 deselected in 288.28s (0:04:48)`，退出码 0。JUnit 时间戳确认两套回归串行执行，汇总为 `summary.json`。

| 验证 | passed | failed | errors | skipped | deselected | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 全量 | 1978 | 0 | 0 | 0 | 0 | 0 |
| real_compile | 80 | 0 | 0 | 0 | 1898 | 0 |
| 定向验证 | 53 | 0 | 0 | 0 | 0 | 0 |

`full-numerical-evidence.json`、`real-numerical-evidence.json` 提取了真实验收输出：两套回归中的 native 探针、FlashAttention、MatmulBiasSiLU 冷 JIT 均为 `source=jit, compiler_calls=1, result=PASS`，两条生产算子的 `toolchain_calls=1`。所有日志均先重定向保存，再读取真实退出码，没有用管道末端命令的退出码替代测试退出码。

## 分发包与安装后验证

实际入口为证据目录中的 `run-packaging.sh`。先备份原 dist，保留已有 build 目录以验证清理，再使用新的 `FLASHINFER_NPU_CACHE_DIR` 且不设置 AOT 模式开关，执行：

```bash
python3 setup.py sdist bdist_wheel
```

构建退出码 0，两条生产 AOT 均为 `disk_cache_hit=false`，汇总 `requested=2, built=2, failed=0, skipped=0`。构建前确实存在旧 `build/lib.linux-x86_64-cpython-312/flashinfer/data/build/kernels`，记录为 `legacy-build-roots.json`；构建后的 wheel 不再包含该树。

`check-packages.py` 对 wheel 执行 ZIP、两条 AOT 完整包校验及 178 个源码/构建输入文件比对；还对 wheel/sdist 的 7 条探针 recipe 全部 SourceRef、摘要、新 schema 位置及旧目录缺失逐项校验。结果保存在 `package-inspection.json`，退出码 0。

| 产物（位于仓库 `dist/`，另备份于证据目录 `wheelhouse/`） | SHA-256 |
| --- | --- |
| `flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl` | `73226ae14f8e48014cf0c2dfcfc44bfa64d594affe39c78b3d2df49311b95249` |
| `flashinfer-0.1.0.dev0.tar.gz` | `33e20b2041114640a39805fe0242ce6e6c108b4f34acdee5880dd679ced2b783` |

随后创建独立 venv，没有使用 `--system-site-packages`；强制安装新 wheel 后才以纯路径 `_deps.pth` 引用 `dy_test` 的依赖。已断言 `flashinfer.__file__` 位于本次 venv，在源码树外执行：

```bash
FLASHINFER_NPU_JIT_DISABLE=1 "$FI_LAYOUT_DIR/venv/bin/python" -B -m flashinfer selftest
"$FI_LAYOUT_DIR/venv/bin/python" -B "$FI_LAYOUT_DIR/installed-probe.py"
```

实际结果：

- selftest 为 `2 PASS, 0 FAIL, 0 SKIP`，退出码 0；两条均命中随包 AOT，并通过 CPU 数值对照。
- 包内 native 探针从全新缓存编译，`disk_cache_hit=false`，退出码 0；使用 `csrc/kernels/ascendc/matmul_probe/native.cpp`，在 NPU 上执行并对照 CPU，`max_abs=0.0`、`result=PASS`。未引用仓库源码目录作为构建根。

完整输出分别为 `package-build.log`、`package-inspection.log`、`wheel-install.log`、`wheel-identity.log`、`selftest.log`、`installed-probe.log`，每条真实退出码均保存在同名 `.exitcode` 文件。编译器原始日志保留在本轮构建缓存中。

完整证据归档为 `../../var/flashinfer/kernel-layout-20261027-LG2E1QlS-evidence.tar.gz`，同名 `.sha256` 保存摘要；归档包含源码改动、报告、wheel/sdist、构建与回归日志、JUnit 及数值记录，排除临时 venv 和开工前 dist 备份。

## 未验证事项

- 其他 SoC、全部 8 卡逐卡数值、下游完整模型集成未在本轮验证。
- 目录迁移不构成图 capture/replay 或性能证据；未修改 `graph_capability`。
- schema 资源只迁移位置，未修改其 schema 版本、枚举或约束。
- 隔离安装包的探针冷编译只执行了 native；CATLASS 探针未单独重复该装后场景。
