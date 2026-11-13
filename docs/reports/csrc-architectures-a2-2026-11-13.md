# 现有 Ascend C 源码按架构拆分与 A2 验收

基线：`46cd939`。本轮将现有五个 `csrc` 算子目录落实为 vLLM-Ascend 风格的架构布局，
保留统一设备入口、公共接口、原生 ABI 与 Python runner。目录划分依据实际实现：
GMM 为 arch22；Matmul、MatmulBiasSilu 与 CATLASS FlashAttention 为 arch35；
原生 FP32 Matmul 探针的通用算法放入 common，其 CATLASS 版本放入 arch35。

## 文件与调用边界

```text
csrc/kernels/
├── gemm/grouped_matmul_swiglu_quant/
│   ├── op_kernel/grouped_matmul_swiglu_quant.cpp       # 统一入口，检查 __CCE_AICORE__ == 220
│   ├── op_kernel/arch22/                              # 原有计算实现、流水线与量化头文件
│   ├── op_host/fi_grouped_matmul_swiglu_quant.cpp       # 原生 OpDef / tiling 绑定入口
│   └── op_host/arch22/                                # 原有平台 tiler
├── gemm/matmul/op_kernel/
│   ├── native.cpp                                    # 统一入口，检查 __CCE_AICORE__ == 310
│   └── arch35/matmul_kernel.h                         # 原有 MMAD / C310 Fixpipe 实现
├── gemm/matmul_bias_silu/op_kernel/
│   ├── native.cpp                                    # 统一入口，保留混合核任务与 tiling ABI
│   └── arch35/matmul_bias_silu_kernel.h                # 原有 Cube / Vector 实现
├── attention/flash_attention/
│   ├── op_kernel/catlass.cpp                          # 统一入口，检查 arch35
│   ├── op_kernel/arch35/                              # 原有 CATLASS FA 实现与工具
│   └── op_host/arch35/fai_tiling.h                     # 原有 Ascend950 tiler
└── gemm/matmul_probe/op_kernel/
    ├── native.cpp                                    # 原有固定三 Tensor 的 v2 入口
    ├── common/native.h                               # 可复用的 FP32 探针算法
    ├── catlass.cpp                                   # CATLASS 入口，检查 arch35
    └── arch35/catlass.h                               # 原有 Ascend950 模板实例
```

参考布局来自 `vllm-ascend: csrc/attention/sparse_flash_attention/op_kernel/arch22/`、
`arch35/` 及其统一设备入口。设备计算按 `__CCE_AICORE__` 选择架构；
host 专用 tiler 跟随其实际平台，共享 OpDef、推形及 host/device 线格式保持公共位置。
缺失架构实现时明确编译失败。Python runner 没有增加架构分支，也没有复制每种产品的接口。

12 个原文件移动后逐字节保持一致；Matmul、MatmulBiasSilu、GMM 和 CATLASS 探针的原生入口
签名及已有计算调用保持原值，原生探针只将其算法提取到公共头文件。
所有生成 recipe/spec 已重新冻结；显式 v2 探针 JSON 刷新源码闭包及摘要，五条身份与 ABI 不变。
生产精确清单和能力快照保持 4 个 Ascend C kernel、85 条全局 Descriptor、47 个逻辑 op；
v1 ABI 测试输入未修改。

关键位置：`csrc/kernels/README.md:27` 说明目录规范；GMM 的 `op_kernel/arch22/`、
`op_host/arch22/` 和 FA 的 `op_kernel/arch35/`、`op_host/arch35/` 保存实际移入文件；
`csrc/recipes/matmul_probe.json:1` 绑定探针完整头文件闭包；
`tests/test_csrc_architectures.py:1` 验证闭包、错误架构拒绝、实际探针运行和 arch35 编译。
接入说明与两个当前设计文档中的 FA 源码链接已同步更新，已有代码块未减少。
设备选择见 `csrc/kernels/gemm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp:16`、
`csrc/kernels/gemm/matmul/op_kernel/native.cpp:4` 和
`csrc/kernels/attention/flash_attention/op_kernel/catlass.cpp:9`；
host 绑定见 `csrc/kernels/gemm/grouped_matmul_swiglu_quant/op_host/fi_grouped_matmul_swiglu_quant.cpp:17`。

## 环境与实际命令

约定解释器 `/root/miniforge3/envs/flashinfer-a5/bin/python` 返回 `Permission denied`，退出码 126。
实际使用下列环境；自检输出 `True 8`，八张卡均为 Ascend910B3，测试选择设备 4。

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
export FLASHINFER_AOT_CACHE_DIR=/tmp/fi-csrc-arch-aot
export FLASHINFER_NPU_CACHE_DIR=/tmp/fi-csrc-arch-cache
export TRITON_CACHE_DIR=/tmp/fi-csrc-arch-triton
export ASCEND_CACHE_PATH=/tmp/fi-csrc-arch-ascend
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/c++/11/backward:/home/dy/tools/conda-envs/megamoe-a2/include
export LIBRARY_PATH=/home/dy/tools/conda-envs/megamoe-a2/lib
/tmp/fi-author-test-env/bin/python -B -m flashinfer.operators generate
/tmp/fi-author-test-env/bin/python -B -m flashinfer.aot --freeze --all
/tmp/fi-author-test-env/bin/python -B -m flashinfer.operators check
```

上述生成、冻结、检查均成功；生成资源仍位于 `build/generated/`。

```bash
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile quick --log-dir /tmp/fi-csrc-arch-ut-01 \
  tests/test_csrc_architectures.py tests/test_operator_authoring.py \
  tests/test_ascendc_v2_public.py tests/test_operator_layout.py
# 38 passed / 7 deselected / 0 skipped; exit_code=0
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile full --log-dir /tmp/fi-csrc-arch-full-01
```

| 范围 | 实际结果 |
| --- | --- |
| 全仓 full | 2091 passed / 44 failed / 63 errors / 142 skipped，共 2340，退出码 1。 |
| 同次 full 中的 real_compile | 44 passed / 44 failed / 34 errors / 0 skipped，共 122；由 JUnit 与标记交叉统计。 |
| 同次 full 中新增源码架构专项 | 11 passed / 0 skipped，含 5 个真实编译用例。 |
| 同次 full 中 GMM 验收 | 6 passed / 0 skipped，包含三种队列模式的数值、缺包和损坏包检查。 |
| 当前设备选择检查 | 7 passed / 0 skipped，退出码 0；探针改为按实际硬件选择目标，兼容后续在 A3/A5 上运行。 |
| 提交前最终架构/GMM 专项 | 17 passed / 0 skipped，含 11 个真实编译用例；退出码 0。 |

最后一项复用了已验证的编译缓存，没有重复冷构建整套矩阵，实际命令为：

```bash
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile full --log-dir /tmp/fi-csrc-arch-final-02 \
  tests/test_csrc_architectures.py::test_current_csrc_native_probe_builds_and_executes_on_available_device \
  tests/test_csrc_architectures.py::test_native_declarations_close_the_selected_architecture_and_platform_tiler \
  tests/test_csrc_architectures.py::test_missing_device_implementation_fails_at_preprocessing
# 7 passed / 0 skipped; exit_code=0
```

提交检查清理了四个新头文件的末尾空行，随后按新的源码摘要重新生成、冻结和验证受影响产物。
最终复验覆盖架构目录闭包、所有架构入口、A2 原生探针、arch35 编译和 GMM 全部六项验收：

```bash
/tmp/fi-author-test-env/bin/python -B tools/run_regression.py \
  --profile full --log-dir /tmp/fi-csrc-arch-final-03 \
  tests/test_csrc_architectures.py tests/test_operator_packages_real.py
# 17 passed / 0 skipped; exit_code=0
```

当前 `csrc` 原生探针的 A2 子进程输出为
`soc=Ascend910B, numerical_cases=2, shared_runner=true, runtime_compilations=0`。
GMM 与该探针共两份 A2 编译产物，以及三份生产和一份探针的 arch35 编译产物，均通过实际产物校验。
arch35 四份产物只验证编译，未在当前 A2 设备上运行。

全仓回归未通过。44 个失败和 63 个错误均落在此前保留的完整回归失败节点中，未出现新增失败节点；
主要涉及 Ascend950 设备要求、目标不匹配，以及旧 host_shared 路径的 `cstdint` 缺失。
142 个 skip 的原因全部是缺少 Ascend950 路径所需设备（16、118、8 三类）。
未放宽测试、修改旧 ABI 输入或将这些 skip 计作通过。
完整日志、JUnit、退出码、产物校验、安装证据及对照基线保存在 `build/csrc-arch-review/`。

## A2 安装运行与交付检查

从本轮编译缓存导出 GMM 的 AOT 包，复制当前源码及生成元数据到隔离构建目录，使用
`FLASHINFER_AOT_PREBUILT` 打 wheel。包内检查的 8 处架构资源全部存在，旧的平铺头文件路径不存在。
使用新的 venv 安装 wheel；没有使用 `--system-site-packages`，安装后才添加纯路径 `_deps.pth`。

```bash
# 在隔离源码目录执行，日志：/tmp/fi-csrc-arch-wheel-build-final.log
FLASHINFER_AOT_PREBUILT=/tmp/fi-csrc-arch-wheel-release-final \
FLASHINFER_ACLNN_ADAPTER_PATH=/tmp/fi-author-adapter \
  /tmp/fi-author-test-env/bin/python -B setup.py bdist_wheel \
  --dist-dir /tmp/fi-csrc-arch-wheelhouse-final
/home/dy/tools/conda-envs/megamoe-a2/bin/python -B -m venv /tmp/fi-csrc-arch-wheel-env
/tmp/fi-csrc-arch-wheel-env/bin/python -B -m pip install --no-deps --force-reinstall \
  /tmp/fi-csrc-arch-wheelhouse-final/flashinfer-0.1.0.dev0-py3-none-linux_aarch64.whl
# 在源码树外执行；bootstrap 只设置设备、校验安装来源并禁止编译。
PYTHONPATH=/tmp/fi-csrc-arch-bootstrap FLASHINFER_TEST_NPU_DEVICE=4 \
FLASHINFER_NPU_JIT_DISABLE=1 \
  /tmp/fi-csrc-arch-wheel-env/bin/python -B -m flashinfer selftest \
  --kernel-id ascendc/gemm/grouped_matmul_swiglu_quant/native/op_package/Ascend910B
# Summary: 1 PASS, 0 FAIL, 0 SKIP (total=1)
```

父子进程均确认 `flashinfer.__file__` 在该 venv 中，并启用禁止工具链发现、原生编译和桥编译的
检查，禁止入口调用记录为 0。GMM 命中包内 spec `8690c8de6ff1`，按原有容差通过独立 CPU
reference 检查；观测最大绝对误差为 INT8 输出 1、scale 约 `7.28e-12`。
此项是 GMM 的单项安装验收，其他三个生产 kernel 的安装运行未由此验证。

本机没有 A3/A5 硬件，相关真机数值、capture/replay 与性能均未验证。
Matmul、MatmulBiasSilu、CATLASS FlashAttention 的生产实现仍只声明 Ascend950；
它们的 A2 移植不在这次源码布局整改中。架构目录存在不构成扩展支持范围的依据。
本轮补齐了上一轮仅在私有样例中展示的架构布局：实际 `csrc` 已分离，并直接验证其原生探针。
