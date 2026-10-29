# Ascend C 通用声明与批量构建验证

源码基线为 `2a71708`，以下结果对应本轮工作树。设备自检实际输出 `True 8`；使用 dy_test 的 Python 3.12、Torch 2.10.0+cpu、torch_npu 2.10.0.post7 和 CANN 9.2.0。

## 范围与实际改动

本轮按用户确认，统一声明、注册和批量入口，保留按 kernel 的缓存与 op_package，不合并 CMake 工程或 vendor 包。

| 位置 | 改动 |
| --- | --- |
| `flashinfer/aot/declarations.py:41` | 从集中 descriptor 与 recipe manifest 按 kernel_id 关联，批量冻结；先校验所选项，再写入声明，不自动扩大 capability/workspace/ABI |
| `flashinfer/jit/dependencies.py:13` | AOT/JIT 共用按依赖名定位的解析器，继续校验固定 revision 与实际头树 |
| `flashinfer/ascendc.py:59`、`:66` | 删除仅识别 CATLASS 的依赖定位，以及按 FA/Matmul 选择冻结脚本的诊断；保留现有 facade 的 kernel_id 常量与无参 load_contract 兼容行为 |
| `flashinfer/aot/build.py:110`、`:232` | 批量构建先完成所有所选项的源闭包/spec 检查；CLI 支持 freeze、重复 kernel-id 和通用 dependency-root |
| `tools/build_flash_attention.py:20`、`tools/build_matmul_bias_silu.py:19` | 旧脚本委托通用框架，只保留存量选择入口；不再分别生成 descriptor 与维护特化字符串 |
| `tests/test_ascendc_framework.py` | 临时新增算子的配置接入、身份/拒绝门禁、批量预检与真实 AOT/JIT 集成测试 |

新增算子提供源代码、recipe 和集中 descriptor 后，不需要再复制一份 Python 构建脚本。公共 facade、Torch schema 和数值验收仍属于算子交付件，不由编译框架自动生成。

## 实际执行的命令

以下 `WORK` 表示仓库外的本轮证据目录；完整日志、JUnit、命令 JSON 和原始退出码保存在该目录，未纳入 Git。

```bash
# 构建环境已设置 FLASHINFER_CATLASS_ROOT，指向固定 revision 的 checkout。

python3 -B -m flashinfer.aot --freeze --all

python3 -B tools/run_regression.py --profile quick \
  --log-dir "$WORK/affected-quick-final" \
  tests/test_ascendc_framework.py tests/test_release_checks.py \
  tests/test_flash_attention_registration.py tests/test_matmul_bias_silu_registration.py \
  tests/test_aot_runtime.py tests/test_ascendc_v2_public.py tests/test_ascendc_recipe.py \
  tests/test_aot_multi_operator.py tests/test_aot_loader_paths.py

export PYTEST_ADDOPTS="--basetemp=$WORK/compile-final-cases"
python3 -B tools/run_regression.py --profile compile \
  --log-dir "$WORK/affected-compile-final" tests/test_ascendc_framework.py

export FLASHINFER_AOT_CACHE_DIR="$WORK/compile-final-cases/test_batch_cli_cold_build_cach0/cache"
python3 -B setup.py build --build-base "$WORK/wheel-build" \
  bdist_wheel --dist-dir "$WORK/wheel"
```

专项 compile 用例依次实际调用通用 AOT CLI 两次，验证冷构建与缓存复用；随后串行运行已有 Matmul/FA 的 numerical 和 missing_enabled 验收脚本。各子命令的完整参数、日志与退出码位于 `compile-final-cases/test_batch_cli_cold_build_cach0/`。

wheel 另装入新建的隔离 venv，未使用 system-site-packages。安装时使用 no-deps/force-reinstall，然后仅用纯路径 `_deps.pth` 提供基础 Torch 依赖，并断言 `flashinfer.__file__` 位于该 venv。装后设置 JIT 禁用、CATLASS 目录不存在、PATH 中无编译器，运行：

```bash
"$WORK/venv/bin/python" -B -m flashinfer selftest
# 在 tests 目录，无参数执行现有脚本；其中启用原生 TorchNPU 桥。
"$WORK/venv/bin/python" -B flash_attention_acceptance.py
```

## 实际结果

| 检查 | 结果 |
| --- | --- |
| 通用批量冻结 | 成功，退出码 0；现有 registry.json 和 ascendc_specs.json 逐字节不变 |
| 临时新增算子 | 仅添加 recipe/集中 descriptor 即可冻结、自动绑定 lazy runner 并准备 JIT backend；此项是契约与接线验证，不代表新增数学算子的真机支持 |
| 受影响 quick | **131 passed、3 deselected、0 failed/errors/skipped**，pytest 3.32 秒，入口 4.688 秒；两个退出码均为 0 |
| 受影响 compile | **1 passed、24 deselected、0 failed/errors/skipped**，pytest 503.98 秒，入口 506.029 秒；两个退出码均为 0 |
| 通用 CLI 冷 AOT | 两条生产项均 disk_cache_hit=false；built=2、failed=0、skipped=0，退出码 0 |
| 同缓存再次构建 | 两条均 disk_cache_hit=true，spec 不变，退出码 0 |
| AOT 数值 | Matmul 8 组 shape，FA 6 组 shape × FP16/BF16，共 12 组；原有容差下通过，编译/工具链调用均为 0 |
| 真实冷 JIT | Matmul 1 组，FA 同一 shape 的 FP16/BF16；各进程实际编译 1 次，重复公共调用复用绑定，全部通过 |
| 默认 AOT wheel | 成功，退出码 0，9.156 秒；两条 NPU 产物均复用已有缓存，并携带原生桥和新增框架模块 |
| 隔离 wheel selftest | **2 PASS、0 FAIL、0 SKIP**，退出码 0 |
| 隔离 wheel Torch FA | 原生桥 functional/out 路径的 FP16/BF16 均通过，退出码 0 |

AOT Matmul 对 FP64 参考的最大绝对误差为 `0.007803267188764451`，FA 对各自参考的最大绝对误差为 `0.0007331222295761108`；具体 shape、容差和逐项误差见各验收 JSON，未放宽既有容差。冷 JIT 的最大绝对误差分别为 `0.0019238364591760426`、`0.0006372779607772827`。

wheel 为 `flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl`，大小 1673192 字节，SHA-256 为 `9051f6254879364ca5271860cbff28b0738682babc479bfae6634af7864615c8`。wheel 内四个新增/修改的框架 Python 模块与当前工作树内容一致。

### 首次集成测试的失败与修正

首次专项执行为 **1 failed、23 deselected**，退出码 1。冷 AOT 与缓存复用已成功，但新增测试将临时 release 搜索路径用于 `selftest`；selftest 有意只检查随包目录，返回 **0 PASS、0 FAIL、2 SKIP**。此结果不计成功。

修正的是新增测试的验证方式：临时产物改用已有公共 API 数值脚本；真正的装后 selftest 在隔离 wheel 环境验证。既有 selftest 实现与测试断言未修改。首次失败日志保留于 `affected-compile/`，最终通过记录位于 `affected-compile-final/`。

## 兼容边界与未验证事项

- 现有生产 kernel_id、descriptor/ABI、recipe、冻结 spec 与设备源码不变；既有测试断言未修改，精确登记集合保持原样。
- 按用户速度要求执行受影响范围，未运行 full 或完整 real_compile 矩阵；专项结果不能作为全量结果。
- 未验证其他 SoC、跨 CANN/Torch 版本、任意 scalar/attribute ABI、生产算子图捕获或完整模型；现有图能力仍为 NOT_CAPTURABLE。
- 多算子同次 CMake 构建、合并 vendor 包不在本轮范围。未做同口径编译性能测量，不声称编译性能提升。

既有结论需要细化：通用 driver/store/loader 不等于完整通用的算子声明工作流。本轮消除了上层专用冻结脚本与依赖定位的接入要求，仍按已声明、已验证的 v2 Tensor 契约构建和执行。
