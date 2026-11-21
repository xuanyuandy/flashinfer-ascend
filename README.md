# flashinfer

**flashinfer** 是面向昇腾 NPU 的推理算子适配框架，在 FlashInfer 风格的 Python API 下集成 CANN、Ascend C 和 Triton 实现。框架根据输入元数据、计算语义、硬件能力和用户配置选择实现，并提供原生算子的构建、加载与交付机制。

[核心特性](#核心特性) · [算子支持](#算子支持) · [后端与硬件](#后端与硬件) · [安装与构建](#安装与构建) · [测试](#测试) · [开发路线](#开发路线)

## 核心特性

- **统一接口与多后端实现**：以逻辑 `op` 关联公共 API、能力声明和后端 runner；提供直接函数、plan/run 与 run-only 接口，按算子契约对接推理框架。
- **基于能力的路由**：将 Tensor 元数据及量化、融合等计算语义映射到标准调用签名，按 dtype、shape、逻辑与物理布局、SoC 和实现约束筛选候选；支持算子自定义属性及合法组合校验。
- **可配置、可诊断的调度**：提供全局策略及接口声明的 `backend` / `kernel_id` 覆盖入口，输出候选拒绝原因和选择依据；有效离线调优记录优先，无记录时按配置、`priority` 和 kernel ID 确定选择。
- **Ascend C AOT 构建与交付**：从算子定义、原生 ABI 和 CMake 配置生成构建元数据与类型化 ACLNN 适配，统一校验源码、依赖、环境和产物身份，并通过 TorchNPU 桥执行。
- **Ascend C 多架构管理**：A2/A3 共用 arch22 源码和 runner，A5 使用 arch35 实现；每个物理 SoC 独立绑定 Descriptor、构建身份和二进制，按芯片分组发布 wheel。
- **声明式算子扩展**：开发者维护公共定义、后端 implementation/runner、设备实现和独立数值样例；框架负责目录发现、索引、摘要、构建缓存和打包，并为带 schema 的直接算子生成 functional/out/FakeTensor 接入。

仅当实现声明与全局策略均允许时，AOT 缺包才进入 JIT；加载、校验和执行异常直接传播。`reference` 为显式选择的参考后端，用于语义与数值对照。路由规则见 [后端路由](docs/concepts/backend-routing.md)，上游 API 的签名差异与适配范围见 [兼容性说明](docs/upstream-compatibility.md)。

## 算子支持

| 算子族 | 实现示例 | 接口与约束 |
| --- | --- | --- |
| Attention | Prefill/Decode、Paged/Ragged KV、CATLASS dense prefill、受限 MLA 场景的 Sparse Flash Attention | [Attention](docs/operators/attention.md) |
| GEMM | FP16/BF16/FP32 MM/BMM、FP8 BMM/分组 GEMM、分段 GEMM、Matmul、MatmulBiasSiLU、GMM + SwiGLU + 量化 | [GEMM](docs/operators/gemm.md) |
| MoE | 融合专家计算、FP8 per-tensor 与 block-scale 实现 | [MoE](docs/operators/gemm.md) |
| Norm / Activation | RMSNorm、LayerNorm、Gemma 变体、SiLU/GELU 门控融合 | [Tensor 算子](docs/operators/tensor-ops.md) |
| RoPE | 位置编码、position ID、cache 与 inplace 形式 | [RoPE](docs/operators/tensor-ops.md) |
| Sampling | Top-K、Top-P、Min-P、联合采样及链式投机采样 | [Sampling](docs/operators/tensor-ops.md) |
| KV cache / 状态合并 | Paged KV append、单状态与多状态合并 | [KV 与状态操作](docs/operators/tensor-ops.md) |

各接口具有独立的 dtype、shape、布局和 SoC 约束，典型组合与对应测试见 [逐算子支持矩阵](docs/backends/overview.md)。Mixed BatchAttention 的默认 NPU provider、完整 MLA/块稀疏 Attention、上游量化转换及通信模块尚未接入；NPU 上的 PDL 参数按兼容契约关闭。

## 后端与硬件

| backend | 实现来源 | 构建方式 |
| --- | --- | --- |
| `cann` | 已安装的 Torch/TorchNPU/CANN API 或组合 | 本框架无设备编译步骤 |
| `ascendc` | 本仓 Ascend C 原生实现，v2 ACLNN op-package | AOT；部分实现允许受控缺包 JIT |
| `triton` | Matmul、Gemma fused Add RMSNorm 的 Triton kernel | JIT |
| `reference` | 显式参考实现 | 随 Python 包提供 |

运行 `python -m flashinfer list-kernels --all` 可逐项查看当前环境的逻辑算子、实现 ID、后端、声明 SoC 和图能力，末行汇总各后端数量。源码显示默认注册表，wheel 中的原生行对应随包目标。设备数值覆盖以对应测试结果为准。

| 产品系列 | 物理 SoC | 源码架构 | wheel 包名 | 原生 AOT 算子示例 |
| --- | --- | --- | --- | --- |
| A2 | `Ascend910B` | arch22 | `flashinfer_ascend910b` | GMM/SwiGLU/量化、SFA、PermuteQkv、UnpermuteQkv、Histgram、MergeChunk2、CATLASS prefill |
| A3 | `Ascend910C` | arch22 | `flashinfer_ascend910b` | GMM/SwiGLU/量化、SFA、PermuteQkv、UnpermuteQkv、Histgram、MergeChunk2、CATLASS prefill |
| A5 | `Ascend950` | arch35 | `flashinfer_ascend950` | Matmul、MatmulBiasSiLU、CATLASS prefill |

A2/A3 共享源码架构与发行包，各自使用严格绑定物理 SoC 的产物。A3 的编译目标和测试入口已登记，真机数值及安装验证尚未完成；CANN/Triton 的支持域仍按具体算子声明判断。

## 安装与构建

### 安装预编译包

选择与目标芯片、主机平台和运行栈匹配的 wheel。NPU 执行依赖匹配的 Torch/TorchNPU/CANN；Triton 实现另需 Triton Ascend。

```bash
python -m pip install ./flashinfer_ascend910b-0.1.0.dev0-py3-none-linux_aarch64.whl
python -m flashinfer --help
python -m flashinfer show-config
```

示例为 A2/A3 的 aarch64 发行包。A5 使用 `flashinfer_ascend950`；Python 导入名均为 `flashinfer`，同一环境安装一个芯片分组。pip 根据主机平台匹配 wheel，NPU 型号由使用方显式选择。AOT-only 实现要求具备匹配的预编译产物；依赖与构建边界见 [后端概览](docs/backends/overview.md#安装与构建边界)。

### 调用示例

以下示例通过 CANN 后端执行矩阵乘，并验证输出。`mm_bf16` 的该支持域覆盖 Ascend910B、Ascend910C 和 Ascend950，使用已安装的厂商算子：

```python
import torch
import torch_npu
import flashinfer

a = torch.ones((16, 32), dtype=torch.bfloat16, device="npu")
b = torch.ones((32, 16), dtype=torch.bfloat16, device="npu")
y = flashinfer.mm_bf16(a, b, backend="cann")

expected = torch.full((16, 16), 32, dtype=torch.bfloat16)
torch.testing.assert_close(y.cpu(), expected)
```

`backend` 和 `kernel_id` 适用于声明这些参数的接口；其他门面的选择方式见 [路由覆盖入口](docs/concepts/backend-routing.md)。

### 源码环境准备

源码构建使用 Python `>=3.9`。在仓库根目录执行下列命令前，激活已安装匹配 Torch/TorchNPU 的环境，并加载 CANN 提供的 `set_env.sh`。Ascend C 编译需要 CANN 开发工具、CMake、C++17 编译器和 libffi 头文件及库；[A5 环境模板](environment-a5.yml) 提供固定的软件依赖组合。

```bash
python -m pip install 'setuptools>=61,<81' wheel 'pytest>=8.3.2,<9'
git submodule update --init --recursive
export FLASHINFER_CATLASS_ROOT="$PWD/3rdparty/catlass"
python -B -c 'import torch,torch_npu;print(torch.npu.is_available(),torch.npu.device_count())'
```

设备自检应返回 `True` 且设备数大于零。CATLASS 使用子模块固定的 v2.2.0，可通过 `FLASHINFER_CATLASS_ROOT` 指定同一提交的外部 checkout。Python 层开发可使用 `FLASHINFER_AOT_ALL=0 python -m pip install -e .` 安装源码模式，原生二进制按需单独构建。

### 原生算子编译与精度验证

以下流程以 A2 的 `sparse_flash_attention` 为例，依次完成元数据生成、一致性检查、设备包构建和 Torch/ACLNN 桥构建：

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.operators check
python -B -m flashinfer.operators build \
  --kernel-id ascendc/attention/sparse_flash_attention/native/Ascend910B \
  --cache-dir build/aot-cache --output-dir build/sfa-aot
python -B -m flashinfer.aclnn.build --output-dir build/aclnn
```

`generate/check` 执行源码发现与元数据生成/核对，不触发设备编译。`build` 先冻结完整原生元数据目录，再编译指定 kernel，因此需要准备目录中声明的全部构建依赖，包括 CATLASS。元数据、设备包和桥分别输出到 `build/generated/`、`build/sfa-aot/` 和 `build/aclnn/`；源码、能力或 CMake 变更后重新生成并构建。类型化 ACLNN 适配由构建系统自动生成，公共接口默认使用该适配。

使用已构建的设备包和桥执行逐输出精度校验，以独立 CPU 计算作为参考：

```bash
FLASHINFER_OPERATOR_TEST_RELEASE="$PWD/build/sfa-aot" \
FLASHINFER_OPERATOR_TEST_ADAPTER="$PWD/build/aclnn" \
python -B tools/run_regression.py --profile full -k sparse_flash_attention tests/operators/test_declared_implementations.py
```

测试通过上述环境变量定位已有设备包和桥；未指定时，由共享测试 fixture 构建本机 SoC 的生产 AOT 集合及桥。该模块标记为 `real_compile`，命令中的 `full` 范围限定为指定文件。用例直接调用 `flashinfer.sparse_flash_attention`，输入组合、独立 CPU 参考计算和容差定义于对应算子的 `cases.py`。

A3 对应 kernel ID 的 SoC 后缀为 `Ascend910C`，复用同一测试模块并在 A3 独立验收。原生 kernel ID 可通过 `python -B -m flashinfer list-kernels` 查询，目标设备与用例分别见 [支持矩阵](docs/backends/overview.md) 和 [精度测试说明](tests/README.md)。运行时通过 `FLASHINFER_AOT_PATHS`、`FLASHINFER_ACLNN_ADAPTER_PATH` 定位设备包与桥；共享测试 fixture 根据复用变量自动配置这两个路径。

`permute_qkv` 复用上述环境和 ACLNN 桥。在 A2 上执行以下命令，可通过 `flashinfer.permute_qkv` 检查 FP16、BF16 的逐元素重排结果。输入为连续物理 ND，输出形状为 `[Nq + 2*Nkv, S, B, D]`。

```bash
python -B -m flashinfer.operators build \
  --kernel-id ascendc/attention/permute_qkv/native/Ascend910B \
  --cache-dir build/aot-cache --output-dir build/permute-qkv-aot
FLASHINFER_OPERATOR_TEST_RELEASE="$PWD/build/permute-qkv-aot" \
FLASHINFER_OPERATOR_TEST_ADAPTER="$PWD/build/aclnn" \
python -B tools/run_regression.py --profile full -k "permute_qkv and not unpermute_qkv" tests/operators
```

`unpermute_qkv`、`histgram` 和 `merge_chunk2` 也可共用一个产物目录和上面的 ACLNN 桥。在仓库根目录生成并检查元数据后，以下命令编译 A2 的三个算子并逐输出比对 CPU 标杆。A3 使用 `Ascend910C` 构建，在对应设备上执行同一组测试。

```bash
soc=Ascend910B
python -B -m flashinfer.operators build \
  --kernel-id "ascendc/attention/unpermute_qkv/native/$soc" \
  --kernel-id "ascendc/fused_moe/histgram/native/$soc" \
  --kernel-id "ascendc/fused_moe/merge_chunk2/native/$soc" \
  --cache-dir build/aot-cache --output-dir "build/qkv-moe-$soc"
FLASHINFER_OPERATOR_TEST_RELEASE="$PWD/build/qkv-moe-$soc" \
FLASHINFER_OPERATOR_TEST_ADAPTER="$PWD/build/aclnn" \
python -B tools/run_regression.py --profile full \
  -k "unpermute_qkv or histgram or merge_chunk2" tests/operators
```

### wheel 构建与安装验证

默认 wheel 构建按本机芯片选择交付分组，编译该分组的完整 AOT 集合并打包原生桥，结果写入 `dist/`：

```bash
python -B setup.py bdist_wheel
```

A2/A3 的 `flashinfer_ascend910b` 包分别交付两种 SoC 的原生产物，A5 使用 `flashinfer_ascend950`。实际打包的 kernel ID 记录在包内 `flashinfer/data/wheel.json`。默认构建采用本机芯片分组；`FLASHINFER_AOT_CONFIG`、`FLASHINFER_AOT_PREBUILT` 和 `FLASHINFER_AOT_ALL=0` 分别提供配置选择、预编译产物打包及源码模式，具体约束见 [构建模式](docs/ascendc_aot.md#原生桥与-wheel)。发布构建可显式指定芯片矩阵：

```bash
python -B -m flashinfer.aot.wheels \
  --soc Ascend910B --soc Ascend910C --soc Ascend950 --output-dir dist
```

安装验证使用独立虚拟环境，预先准备匹配的 Torch/TorchNPU，并在源码目录外执行导入检查与数值自检。以下采用 A2/aarch64 文件名，版本和平台标签以 `dist/` 中的实际产物为准：

```bash
python -m pip install --force-reinstall ./dist/flashinfer_ascend910b-0.1.0.dev0-py3-none-linux_aarch64.whl
cd ..
python -B -c 'import sys,flashinfer;from pathlib import Path;print(flashinfer.__file__);assert Path(flashinfer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())'
python -B -m flashinfer selftest
```

`selftest` 依据当前物理 SoC 选择随包原生实现，在禁用 JIT 的条件下与 CPU 参考结果比较，无需源码测试目录。其覆盖范围由各算子的自检样例定义；完整 dtype/shape 矩阵使用源码精度测试或 [通用算子验收](docs/development/testing.md#通用算子验收)。各物理 SoC 分别执行验收并记录结果。

## 测试

测试分为算子精度、框架行为和构建交付三类，分别位于 `tests/operators/`、`tests/framework/` 和 `tests/packaging/`。在仓库根目录按验证目标选择执行范围：

| 目的 | 命令 | 范围 |
| --- | --- | --- |
| 日常快速回归 | `python -B tools/run_regression.py --profile quick` | 排除 `real_compile`，仍可能使用 NPU |
| 算子精度 | `python -B tools/run_regression.py --profile full tests/operators` | 按本机 SoC 运行算子用例，原生用例准备 AOT 产物 |
| 编译与交付专项 | `python -B tools/run_regression.py --profile compile tests/packaging/test_operator_packages_real.py` | 仅执行指定范围的编译、产物与执行检查 |
| 本机完整回归 | `python -B tools/run_regression.py --profile full` | 包含适用的编译用例，无需随后重复同一编译矩阵 |

回归入口默认按本机 SoC 筛选，将日志、JUnit、原始退出码和结果摘要保存至 `build/regression/`。skip 或零用例对应非零退出码；不适用的其他芯片用例记录为 `deselected`，验证范围单独列出。精度测试、产物复用和验收条件见 [测试说明](tests/README.md) 与 [验收规范](docs/development/testing.md)。

## 开发与扩展

公共算子定义位于 `flashinfer/operators/<domain>/<op>/`，后端实现位于 `flashinfer/<backend>/<domain>/<op>/`。Ascend C 计算源码位于 `csrc/kernels/`，Triton kernel 与其后端实现共同维护。新增算子通过公共定义描述语义，通过各实现声明支持域与执行入口，并提供独立数值样例；生成元数据和交付资源由框架管理。

实现分层与文件职责见 [架构](docs/concepts/architecture.md)，完整接入流程见 [算子开发指南](docs/development/adding-an-operator.md)，维护与审计入口见 [仓库工具](tools/README.md)，提交规范见 [贡献约定](CONTRIBUTING.md)。

## 开发路线

| 方向 | 当前基础 | 扩展目标 |
| --- | --- | --- |
| 图模式 | 受限 CANN single-decode 参数更新接口；生产 Ascend C 实现保持 `NOT_CAPTURABLE` | 原生 AOT 图执行、资源准备复用、更多 Attention 场景的 capture/replay 验证 |
| 调优 | 显式离线测量、记录存储与查询，路由复用有效结果 | 通用 AutoTuner 执行上下文、tactic 搜索及完整执行身份失效管理 |
| 接口与后端一致性 | 部分 FlashInfer API 适配、统一逻辑 op 与多后端路由 | 补齐模型工作负载所需的参数与量化组合、等价后端及框架集成验证 |

设计范围、验收条件与兼容性差距见 [特性设计](docs/features/README.md)、[开发规划](docs/roadmap.md) 和 [上游兼容性](docs/upstream-compatibility.md)。

## 文档

[开始使用](docs/getting-started.md) · [架构](docs/concepts/architecture.md) · [路由](docs/concepts/backend-routing.md) · [算子支持](docs/backends/overview.md)

[接入算子](docs/development/adding-an-operator.md) · [编译交付](docs/ascendc_aot.md) · [测试验收](docs/development/testing.md) · [图模式](docs/npu_graph_extensions.md) · [Tuning](docs/tuning.md) · [规划与差距](docs/roadmap.md)

完整说明见 [文档导航](docs/README.md)；上游签名与能力边界见 [兼容说明](docs/upstream-compatibility.md)。

## 贡献、许可与致谢

协作规则见 [贡献约定](CONTRIBUTING.md)。项目使用 [Apache-2.0](LICENSE)；移植代码的版权头、[集中来源声明](THIRD_PARTY_NOTICES.md) 与 [CANN 开源许可](LICENSES/CANN-Open-Software-License-2.0.txt) 随代码保留。

本仓的 API 分类与调用风格参考 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。上游致谢：FlashInfer is inspired by [FlashAttention](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598), [CUTLASS](https://github.com/nvidia/cutlass), and [AITemplate](https://github.com/facebookincubator/AITemplate).

## Citation

FlashInfer 上游论文：

```bibtex
@article{ye2025flashinfer,
    title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
    author = {
      Ye, Zihao and
      Chen, Lequn and
      Lai, Ruihang and
      Lin, Wuwei and
      Zhang, Yineng and
      Wang, Stephanie and
      Chen, Tianqi and
      Kasikci, Baris and
      Grover, Vinod and
      Krishnamurthy, Arvind and
      Ceze, Luis
    },
    journal = {arXiv preprint arXiv:2501.01005},
    year = {2025},
    url = {https://arxiv.org/abs/2501.01005}
}
```
