# 普通 wheel 构建默认携带生产 AOT（2026-10-27）

## 问题与改动

用户删除 `build/`、`dist/` 后执行 `python3 setup.py bdist_wheel`，安装所得 wheel，再无参数运行 FlashAttention 脚本，得到 `AotMissingError`：包内没有 AOT 候选，脚本禁用 JIT，因此无法执行。原先普通构建只打包源码；这个错误不能据此判断为设备编译失败。

本次将普通 wheel 构建改为调用既有生产 AOT builder，默认编译生产登记表中的两条 Ascend C 实现：native MatmulBiasSiLU 和 catlass FlashAttention。host 库、设备二进制、manifest 一并入包；依赖、摘要、工具链和包完整性校验仍由原有实现执行。CANN 算子使用安装好的运行时，Triton 路径沿用已有编译方式；本次不宣称全部公共接口都生成了 AOT 二进制。

关键位置：

- `setup.py:20`：未设置构建模式时默认选择全部生产 AOT；显式 `FLASHINFER_AOT_ALL=0` 或空值保留源码 wheel 模式。显式构建模式之间仍互斥。
- `setup.py:41`：构建时优先使用显式 catlass 路径，否则查找 `3rdparty/catlass`、`../catlass-ref`。没有新增运行时探测。
- `setup.py:88`：使用原有 builder/store 打包完整产物，保留平台 wheel 标签及复用构建目录时的清理逻辑。
- `flashinfer/data/kernels/ascendc_specs.json:129`、`:341`：适配用户 `dy_test` 构建工具链，更新两条 spec 的名称、环境指纹及 resolved recipe 身份，共 6 行；ABI、源码摘要、recipe、descriptor 及特化范围均未改，`registry.json` 字节未变。
- `tests/flash_attention.py:8`、`tests/matmul_bias_silu.py:8`：无参数调用已安装 wheel，生成 torch Tensor，执行公共算子接口并与 CPU FP64 结果比较；禁用 JIT，检查导入路径位于当前环境。
- `tests/flash_attention_acceptance.py:248`、`tests/matmul_bias_silu_acceptance.py:172`：无参数时运行上述脚本，带参数时保留原有验收入口。
- `tests/release_checks/wheel_checks.py:33`：原有完整输出、精确集合及清理断言同时验证默认模式和显式 ALL 模式；源码模式测试改为显式选择 `0`。
- `tests/__init__.py:1`、`tests/test_aot_discovery_index.py:22`、`tests/test_ascendc_contract_v2.py:21`：明确本仓测试包及其辅助模块的导入路径，避免用户环境中的第三方同名 `tests` 包覆盖。
- `flashinfer/attention/numerics.py:114`、`flashinfer/attention/reference.py:865`：标量参考实现显式逐项相加，保持原有 v1 归约顺序，避免 Python 3.12 内置浮点求和影响固定 corpus 身份；没有修改 corpus 指纹断言、数值策略常量或 NPU 算子源码。

操作说明增量更新于 `README-wheel-conda.md`，没有重写既有设计文档或历史回归报告。

## 实测环境

分支 `infer`，HEAD `3ece69a`，在已有迁移修复的工作区继续实施。本机没有交接材料提到的 `5060a5c`，不据此声称验证了该提交。

| 项目 | 本次实际值 |
| --- | --- |
| conda 环境 | `../../tools/conda-envs/dy_test` |
| Python | 3.12.16 |
| 环境脚本 | `../../tools/miniconda3/Ascend/ascend-toolkit/set_env.sh` |
| CANN | 9.2.0，`../../tools/miniconda3/Ascend/cann-9.2.0` |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post7` |
| Triton Ascend / pytest / NumPy | `3.6.0` / `8.4.2` / `1.26.4` |
| golden 依赖 | 为本轮回归安装 `ml-dtypes==0.5.4`，退出码 0 |
| catlass | `../catlass-ref`，revision `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| 工具链指纹 | `226cecda7bb975ee0ba3e39f449aac371f1ebc7cdcba90d6198678804884d250` |
| 设备自检 | `True 8` |

实际激活 `dy_test`、加载用户指定 CANN 环境脚本后执行：

```bash
python3 -B -c 'import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())'
```

输出 `True 8`，退出码 0。完整环境版本及工具链路径保存在 `build-environment.log`、`pip-freeze.txt`。

## 从空构建目录执行普通打包

本次证据目录为 `../../var/flashinfer/default-aot-20261027-uyhT1qSw/`，下文日志文件名相对于该目录。该目录在仓库 `build/`、`dist/` 之外，删除这两个输出目录不会删除本轮日志。

将旧 `build/`、`dist/` 移入证据目录的 `previous-build/`、`previous-dist/` 备份，使用全新缓存；取消显式 AOT 模式和 catlass 路径变量，以检验默认行为和相邻 checkout 发现：

```bash
unset FLASHINFER_AOT_ALL FLASHINFER_AOT_CONFIG FLASHINFER_AOT_PREBUILT
unset FLASHINFER_AOT_SOCS FLASHINFER_AOT_CACHE_DIR FLASHINFER_CATLASS_ROOT
unset FLASHINFER_NPU_JIT_DISABLE
export FLASHINFER_NPU_CACHE_DIR="$FI_TASK_DIR/cold-cache"
python3 setup.py bdist_wheel
```

`FI_TASK_DIR` 为上述证据目录。实际 stdout/stderr 完整保存于 `default-wheel-build.log`，在读取日志前保存真实退出码至 `default-wheel-build.exitcode`，结果为 0。构建汇总：

```text
MatmulBiasSiLU: disk_cache_hit=false
FlashAttention: disk_cache_hit=false
{"summary": {"built": 2, "failed": 0, "requested": 2, "skipped": 0}}
```

两条编译过程的完整构建日志位于 `cold-cache/` 内，包含 CMake/CANN 实际执行记录。首次冷构建 wheel 为 `flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`，另存于证据目录的 `initial-wheelhouse/`，大小 1,572,394 字节，SHA-256：

```text
d8c368e48ba3f47e589eb83d9ea241e25b21afced9debf378c372e0b0be1447b
```

对 wheel 执行 ZIP 完整性、两个 manifest 的精确数量、解包后的 build store/op-package 校验。两个设备文件均为 NPU ELF，`ET_EXEC`、machine `0x1029`，不是仅有 Python 源码或 host 空壳：

| 生产算子 | spec | 设备二进制字节数 |
| --- | --- | --- |
| MatmulBiasSiLU | `f7d5c89206b85d3434b0804e2ca6133936826eacdd5bca123dfb6662c09d2e19` | 11,592 |
| FlashAttention | `a2185796818cf182ebc3b88754a073fcd11d17c1bac8d301d9482c9a5e6bf98e` | 29,992 |

详情见 `wheel-inspection.json`。后续标量参考兼容修复后重新打包的最终 wheel 信息见下文。

## 安装后直接调用

使用 `dy_test`，在源码目录外强制安装本次 wheel，退出码 0；随后进入仓库 `tests/`，逐条执行：

```bash
python3 -m pip install --force-reinstall --no-deps "$FI_WHEEL"
python3 flash_attention.py
python3 flash_attention_acceptance.py
python3 matmul_bias_silu.py
python3 matmul_bias_silu_acceptance.py
```

`FI_WHEEL` 指向本次生成的 wheel。四个脚本均无参数运行，退出码均为 0，打印的 `flashinfer` 路径为 `../../tools/conda-envs/dy_test/lib/python3.12/site-packages/flashinfer/__init__.py`。两个文件名入口分别得到相同结果：

```text
output: (128, 2, 128) torch.float16
max_abs: 6.122825770479845e-05
PASS flash_attention

output: (16, 32) torch.bfloat16
max_abs: 0.003581114843043798
PASS matmul_bias_silu
```

另在源码树外执行 `python3 -m flashinfer selftest`，退出码 0；两条均为 `bundled AOT hit; compilation=disabled; CPU reference matched`，汇总：

```text
Summary: 2 PASS, 0 FAIL, 0 SKIP (total=2)
```

原始输出及退出码分别保存于同名 `.log`、`.exitcode` 文件，安装记录为 `install.log`、`install.exitcode`。这些结果覆盖本轮样本，不表示所有输入均已数值验证。

## Python 3.12 回归失败与最终 wheel

初次全量执行为 `1977 passed, 1 failed, 0 errors, 0 skipped`，退出码 1。唯一失败为 `tests/test_attention_corpus.py:19` 的固定指纹断言：Python 3.11 得到既有 `7577420e…b38da536`，Python 3.12 得到 `bd811638…5a89322`。在 Python 3.12 进程中仅把参考实现的求和替换为逐项累加，就恢复原指纹；不是改写期望值。

将标量参考实现的加权输出及 softmax 分母改为显式累加后，两种解释器均得到完整原指纹 `7577420ea94a1c97352ea2265a76939b8c566aeaf792daf083a123d3b38da536`，各自 14 个 replay 通过，退出码均为 0。另执行：

```bash
python3 -B -m pytest -p no:cacheprovider -q \
  tests/test_attention_corpus.py tests/test_attention_numerics.py tests/test_attention_reference.py
```

结果 `37 passed in 0.17s`，退出码 0，见 `oracle-targeted.log`。原指纹测试文件没有改动。

修复前尚未完成的 `real_compile` 被主动中止，实际输出为 `1 passed, 1898 deselected`、`KeyboardInterrupt`，退出码 2；这是未完成的验证，不计为通过。它与该轮全量日志一起保存在 `before-oracle-fix/`。

通过 `run-final-wheel.sh` 再执行普通 `python3 setup.py bdist_wheel`，退出码 0。NPU 构建输入未改，这次使用首次冷编译的已验证产物，两条 `disk_cache_hit=true`，仍为 `requested=2, built=2, failed=0, skipped=0`。最终 wheel 位于 `dist/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`，并备份于证据目录的 `wheelhouse/`，大小 1,572,538 字节，SHA-256：

```text
a9e309efde558e7e07121771fee1897207fe50c4e8ecdd4779b3d07b0f12a7aa
```

最终 wheel 的两个 manifest、完整包校验、NPU ELF 类型及 177 个源码/构建输入均重新检查；设备二进制与首次冷编译逐字节相同。见 `final-wheel-inspection.json`、`change-review.json`。

最终 wheel 强制重装后，四个无参数脚本再次全部 PASS，selftest 再次为 `2 PASS, 0 FAIL, 0 SKIP`，每条退出码均为 0，误差与首次安装结果相同。日志加 `final-` 前缀保存。另在源码树外使用最终安装包生成 corpus，原固定指纹及 14 个 replay 均通过，见 `final-installed-corpus.log`，退出码 0。

更早的两轮收集失败分别保存在 `first-collection/`、`second-collection/`：第一轮因第三方 `tests` 包遮蔽本仓而出现 43 个收集错误；加入包声明后，第二轮暴露两处未使用包名的辅助模块导入，出现 2 个收集错误。两轮全量与 `real_compile` 的退出码均为 2，没有将其计为通过。修正导入后才获得上述完整的 1978 例执行结果。

## 最终串行回归

实际入口为证据目录中的 `run-regressions.sh`。在 `dy_test` 加载 CANN，设置本机测试解释器与 catlass checkout，并清除禁编译及外部 AOT/vendor 搜索覆盖后，串行执行以下两条命令；先保存 `$?`，再读取日志：

```bash
python3 -B -m pytest -p no:cacheprovider -q \
  -o junit_logging=all -o junit_log_passing_tests=true \
  --basetemp="$FI_TASK_DIR/full-temp" --junitxml="$FI_TASK_DIR/full.xml" \
  > "$FI_TASK_DIR/full.log" 2>&1
FI_FULL_RC=$?
printf '%s\n' "$FI_FULL_RC" > "$FI_TASK_DIR/full.exitcode"

python3 -B -m pytest -p no:cacheprovider -q -m real_compile \
  -o junit_logging=all -o junit_log_passing_tests=true \
  --basetemp="$FI_TASK_DIR/real-temp" --junitxml="$FI_TASK_DIR/real.xml" \
  > "$FI_TASK_DIR/real.log" 2>&1
FI_REAL_RC=$?
printf '%s\n' "$FI_REAL_RC" > "$FI_TASK_DIR/real.exitcode"
```

全量实际输出为 `1978 passed, 4 warnings in 312.28s (0:05:12)`，退出码 0；JUnit 确认 `0 failed, 0 errors, 0 skipped`。4 条 warning 为 torch_npu 基础格式提示及 3 条 Triton 弃用提示，完整原文保留在日志中。

独立 `real_compile` 输出为 `80 passed, 1898 deselected in 256.48s (0:04:16)`，退出码 0。最终结果如下，JUnit 时间戳确认串行执行，汇总见 `summary.json`：

| 验证 | passed | failed | errors | skipped | deselected | 退出码 |
| --- | --- | --- | --- | --- | --- | --- |
| 全量 | 1978 | 0 | 0 | 0 | 0 | 0 |
| real_compile | 80 | 0 | 0 | 0 | 1898 | 0 |
| 标量参考定向测试 | 37 | 0 | 0 | 0 | 0 | 0 |
| 最终 wheel selftest | 2 | 0 | — | 0 | — | 0 |
| 最终 wheel 无参数脚本 | 4 | 0 | — | 0 | — | 0 |

`full-numerical-evidence.json`、`real-numerical-evidence.json` 从通过用例的 JUnit stdout 提取了真实数值验收记录。两套回归中的 native probe、FlashAttention、MatmulBiasSiLU 冷 JIT 均记录 `source=jit, compiler_calls=1, result=PASS`；两条生产算子的 `toolchain_calls` 均为 1。本次包含实际编译和执行。

完整证据另归档于 `../../var/flashinfer/default-aot-20261027-uyhT1qSw-evidence.tar.gz`，SHA-256 随同名 `.sha256` 文件保存。归档包含最终 wheel、首次冷编译日志、最终及历史回归日志/退出码、JUnit、数值记录、源码改动和本报告；不收录开工前备份的旧 `build/`、`dist/`。

## 校验边界与未验证事项

修改过的测试文件中，HEAD 的 175 条 `assert` 表达式均保留，新增 5 条；具体 AST 核对见 `change-review.json`。源码 wheel 清理测试的环境输入改为显式 `0`，仍严格断言产物目录不存在、输出列表为空、wheel 无 native 标记。文档代码块没有减少，未搬动章节。

- 旧服务器 `-9/SIGKILL` 的信号发送者和根因仍未验证；不能由本机编译成功反推旧故障原因。
- 多 SoC、全部 8 卡逐卡数值、新建 conda 环境中的本轮 wheel 调用，均未在本轮重做；本轮实际使用用户已有的 `dy_test`。
- 图 capture/replay、性能比较及下游完整模型集成未验证；没有改动 `graph_capability`。
- 本轮 wheel 直接脚本与 selftest 覆盖两条生产 Ascend C 实现；不能外推为所有公共接口的装后验证或全量 Triton AOT 支持。
