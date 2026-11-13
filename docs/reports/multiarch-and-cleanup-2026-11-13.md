# 多目标算子交付与冗余文件清理

基线：`516f4b2`。同一逻辑算子、同一 native ABI 可以复用 runner；框架按目标生成
独立 Descriptor、recipe、冻结 spec 和 AOT 包。此次通过私有矩阵乘探针验证框架链路，
不把现有生产算子的支持范围扩展到未经验证的产品。

## 改动与职责

| 位置 | 作用 |
| --- | --- |
| `flashinfer/ascendc/definition.py:80` | `for_targets()` 一次声明公共构建配置、ABI 和 runner，展开 A2/A3/A5 变体；生成文件按目标区分。 |
| `flashinfer/operators/authoring.py:185` | 修复同一模块只处理第一条原生声明的问题；保留模块批量生成入口与输出冲突校验。 |
| `flashinfer/jit/op_package.py:43` | 补齐 `Ascend910C` 到 CANN `ascend910_93`、`dav-2201` 的映射。 |
| `flashinfer/runtime/hardware.py:50` | 将 TorchNPU 的 `Ascend910_93xx` 设备名归一化为 A3 系列 `Ascend910C`。 |
| `flashinfer/operators/testing.py:62` | 通用验收先过滤当前目标，之后仍要求恰好一个 AOT 实现；构建、损坏包及子进程验收复用同一选择。 |
| `tests/multiarch_case.py:1` | 私有 vLLM 风格样例：一个设备入口、`arch22/arch35`、共享 OpDef 与真实标准 Tensor runner。 |
| `tests/test_multiarch_authoring.py:1` | 验证三个精确 ID、独立 recipe、统一 ABI、头文件闭包、错误目标拒绝、三目标编译与当前设备数值。 |
| `docs/authoring-and-resources.md:82` | 文件树、产品映射、声明范例、可选 CMake 和 runner 复用边界；主设计及接入指南增加入口。 |

设备代码沿用 `vllm-ascend: csrc/attention/sparse_flash_attention/op_kernel/` 的组织方式，
在算子入口按 `__CCE_AICORE__` 选择 `arch22/arch35`。host tiling 负责不同平台的资源差异，
可选算子 CMake 按 `ASCEND_COMPUTE_UNIT` 配置编译选项。公共 Python 接口和 runner 不按架构分支；
若 Tensor/属性 ABI 或能力不同，则分别声明 binding/实现。

清理共 26 个文件：23 份过期任务/交接文档，以及三个私有 Triton 转发文件。
两个 JIT 示例改为直接引用算子目录内的真实 kernel 源文件，文档链接同步更新。
历史报告保留其基线含义，旧任务文档可从 Git 历史恢复。
仍有消费者的 `flashinfer/jit/attention.py`、v1 ABI 测试输入与当前构建模板继续保留。
保留文档的代码围栏没有减少：接入指南 28→28，主设计 64→64，作者说明 4→8。

精确生产清单、能力快照与 ABI 常量未改：4 个生产 Ascend C kernel、85 条全局 Descriptor、
47 个逻辑 op、1 个实验声明。多目标探针位于测试隔离项目，不进入生产发现目录。

## 环境与实际命令

约定解释器 `/root/miniforge3/envs/flashinfer-a5/bin/python` 返回 `Permission denied`，退出码 126。
实际使用下列环境，设备自检输出 `True 8`；八张卡均为 `Ascend910B3`，测试选择设备 4。
未加载 CANN 环境时曾出现 `libhccl.so` 缺失，加载后自检正常。

```bash
source /home/dy/tools/miniconda3/Ascend/ascend-toolkit/set_env.sh
/tmp/fi-author-test-env/bin/python -B -c \
  'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count());print([torch.npu.get_device_name(i) for i in range(torch.npu.device_count())])'
# True 8
# ['Ascend910B3', 'Ascend910B3', 'Ascend910B3', 'Ascend910B3',
#  'Ascend910B3', 'Ascend910B3', 'Ascend910B3', 'Ascend910B3']
export FLASHINFER_CATLASS_ROOT=/tmp/flashinfer-catlass-ref
export FLASHINFER_TEST_NPU_DEVICE=4
export FLASHINFER_OPERATOR_TEST_ADAPTER=/tmp/fi-author-adapter
export FLASHINFER_NPU_CACHE_DIR=/tmp/fi-multiarch-cache
export FLASHINFER_MULTIARCH_CACHE_DIR=/tmp/fi-multiarch-builds
export TRITON_CACHE_DIR=/tmp/fi-multiarch-triton
export ASCEND_CACHE_PATH=/tmp/fi-multiarch-ascend
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/c++/11/backward:/home/dy/tools/conda-envs/megamoe-a2/include
export LIBRARY_PATH=/home/dy/tools/conda-envs/megamoe-a2/lib
```

生成、导入及现有四份契约的验证实际输出分别为 `PASS`、`modules=573/failing=0`，
以及 `4 frozen contracts: current toolchain and closed sources match; no compilation`。

```bash
/tmp/fi-author-test-env/bin/python -B -m flashinfer.operators generate
/tmp/fi-author-test-env/bin/python -B -m flashinfer.operators check
/tmp/fi-author-test-env/bin/python -B -m tools.check_imports
/tmp/fi-author-test-env/bin/python -B -m flashinfer.aot --freeze --all
```

实际回归命令如下；`full` 加具体文件只表示该文件范围包含编译用例，不是全仓结果。

```bash
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile full --log-dir /tmp/fi-multiarch-full-01
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile full --log-dir /tmp/fi-multiarch-targets-02 \
  tests/test_multiarch_authoring.py tests/test_hardware.py
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile quick --log-dir /tmp/fi-multiarch-ut-04 \
  tests/test_multiarch_authoring.py tests/test_hardware.py tests/test_operator_authoring.py \
  tests/test_author_api.py tests/test_ascendc_op_package.py tests/test_operator_layout.py
FLASHINFER_AOT_CACHE_DIR=/tmp/fi-multiarch-gmm \
  /tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile compile --log-dir /tmp/fi-multiarch-gmm-03 tests/test_operator_packages_real.py
```

| 范围 | 实际结果 |
| --- | --- |
| 全仓 full | 2067 passed / 18 failed / 98 errors / 142 skipped；退出码 1。 |
| 上述 full 的 real_compile 子集 | 30 passed / 18 failed / 69 errors / 0 skipped，共 117；由 JUnit 与用例标记交叉统计。 |
| 多目标与硬件专项 | 48 passed / 0 skipped，含 1 个真实编译用例；退出码 0。 |
| 最终受影响 quick | 112 passed / 4 deselected / 0 skipped；退出码 0，未执行编译用例。 |
| 重新冻结后的 GMM 编译验收 | 6 passed / 0 skipped；退出码 0，覆盖三种队列数值、缺包与损坏包。 |

全仓原始失败不能记为成功：98 个 setup errors 中 39 个是本轮标准头文件配置与旧冻结
环境身份不一致，59 个明确要求 Ascend950；142 个 skip 均要求 Ascend950（16/118/8 三类原因）。
其余失败涉及 Ascend950 目标在当前 A2 上被拒绝、旧 host_shared 路径的 `cstdint` 缺失，
以及本次探针输出参数名称错误。探针的 binding 现已与 OpDef 的 `c` 保持一致，未放宽 ABI 校验；
修正后的三目标专项通过。四份生产声明已按当前环境重新冻结并核对源码闭包，随后定向复验 GMM。
没有重复整套编译矩阵，也没有把定向复验合并成一份不存在的全绿 full 结果。
原始日志、JUnit、退出码、摘要及探针输出保存在 `build/multiarch-review/`，不进入源码清单。

## 三目标证据与未验证项

| 探针目标 | 实际编译/导出 | 实际数值运行 |
| --- | --- | --- |
| A2 / Ascend910B / arch22 | 成功，spec `c102e083cce6` | Ascend910B3：functional/out 两次检查通过。 |
| A3 / Ascend910C / arch22 | 成功，spec `048dd72b06cb` | 未验证：当前无 A3 设备。 |
| A5 / Ascend950 / arch35 | 成功，spec `10a014a18c74` | 未验证：当前无 A5 设备。 |

探针实际经过公共执行编排、dispatcher、生产 `bind_kernel()`、AOT 索引与标准 Tensor runner。
子进程输出 `shared_runner=true, numerical_cases=2, runtime_compilations=0`，只绑定当前设备的产物。
头文件修改会使三份 recipe 失效；单独选择 A3 时也会正确生成对应文件。

本轮不新增性能或 capture/replay 结论；未重做干净 wheel 安装后的 selftest。
现有 GMM、Matmul、FlashAttention 的生产能力未因探针通过而扩大。
此前提出的 SparseFlashAttention 多后端迁移未在此补丁中登记，仍需独立接入和数值验收。

修正的既有判断：一个 `implementation.py` 允许多条声明，不代表旧生成器已完整处理所有目标；
已有 A3 系列名也不代表已经识别 TorchNPU 的 `Ascend910_93xx` 名称。此次分别补齐生成与归一化。
