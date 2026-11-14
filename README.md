# FlashInfer Ascend

FlashInfer Ascend 是面向昇腾 NPU 的推理算子库与 kernel 接入框架，参考 FlashInfer
的 API 分类、直接函数和有状态 plan/run 接口，接入 CANN、Ascend C、Triton 等实现。
项目仍在开发中，各算子的 dtype、shape、布局、设备和执行模式以具体契约及验证记录为准。

[核心功能](#核心功能) · [算子范围](#算子范围) ·
[快速开始](#快速开始) · [测试与验证](#测试与验证) · [文档导航](#文档导航)

编译 AOT wheel、创建独立 conda 环境并直接运行测试脚本，见
[wheel 编译与 conda 安装指南](README-wheel-conda.md)。未设置其他构建模式时，
普通 wheel 构建会编译并打包当前登记的五条生产 Ascend C 算子。

完整接口清单、上游调用方式及实现差距见 [算子覆盖对比](docs/operator-coverage.md)
和 [公共接口生命周期与测试要求](docs/operator-call-patterns.md)。
不同接口有不同调用方式，直接函数、plan/run wrapper 和 run-only wrapper 均有对应实现。

## 核心功能

- **FlashInfer 风格的 Tensor API**：single attention、GEMM、归一化、RoPE、采样等提供
  直接函数；paged/ragged batch attention 提供 plan/run wrapper；
  `SegmentGEMMWrapper` 通过 `run()` 接收分段元数据。
- **多后端选择**：基于输入 signature、硬件 signature 和 kernel registry 过滤能力，
  支持显式指定 `backend`、`kernel_id` 并解释选择结果。调优层位于实现选择层，
  可使用有效的测量记录；当前随包调优记录为空，未命中时使用确定性策略，不在算子调用中现场测量。
- **通用 Ascend C 编译框架**：从类型化后端声明生成 descriptor 与 recipe，按 `kernel_id` 关联，统一冻结声明、
  解析依赖和批量构建。普通 wheel 随包提供生产 AOT；运行期优先加载已验证 AOT，
  缺失时按策略使用 JIT。缓存和产物仍按 kernel 保存，损坏或加载失败明确报错。
- **目录式算子接入**：作者维护公共 `operator.py`、后端 `implementation.py/runner.py`、可选 CMake、`cases.py` 与 host/device
  实现；`python -m flashinfer.operators build` 自动生成声明/摘要、冻结并构建 AOT。
  公共导出、Torch functional/out/FakeTensor 和装后检查使用通用接线，见
  [作者文件与 CMake 配置](docs/adding-a-custom-operator.md#15-作者文件自动生成项与共有框架)。
- **统一 TorchNPU / ACLNN 执行**：FA 与 MatmulBiasSiLU 的 Python 公共函数是自动注册的
  薄封装，统一进入 `torch.ops.flashinfer.*`，再经 C++ `OpCommand` 桥调用 ACLNN。
  提供 functional/out schema 与 FakeTensor 元数据实现；`enable_aclnn()` 仅做幂等注册，
  已移除默认 Python ctypes 执行分支。
- **可扩展 dtype 契约**：公开 Ascend C recipe 使用 `schema_version=2`，可用
  `dtype_signatures` 声明按参数顺序排列的完整合法 dtype 组合。FA 已验证 FP16/BF16
  的编译、打包、分发和装后执行，未声明的混合组合明确拒绝。
- **开发与交付工具**：产物完整性校验、装后 selftest、保存日志/JUnit/退出码的分档回归，
  以及算子和公共接口清单。

五条生产 Ascend C 算子的统一调用链为：Python 薄封装或直接 Torch 调用 → Torch schema /
PrivateUse1 实现 → signature / dispatch → AOT/JIT → 已验证 Tensor runner →
C++ TorchNPU `OpCommand` → ACLNN host tiling / workspace / launch → device kernel。
CANN 内置算子与 Triton 保留各自的执行路线。逐层职责见 [FA 完整调用链](docs/flash-attention-call-chain.md)。

CATLASS AOT 与新增 Triton JIT FA 可在独立 registry 中比较，并通过同一个 FA 接口使用
精确 shape 调优记录；默认生产登记集合保持不变。使用方法、同步执行边界与无显著差异时的
回退规则见 [FA 隔离调优](docs/runtime-and-compilation-design.md#94-catlass-aot-与-triton-jit-的隔离调优)。

## 算子范围

当前运行时登记表包含 **48 个逻辑 op、86 条实现描述（descriptor）**，其中
**46 个 op 至少绑定一个 NPU runner**；内部 `attention.decode`、`attention.prefill`
仅有 reference 实现。86 条描述由 41 条 CANN、5 条 Ascend C、2 条 Triton 和 38 条 reference 组成。
这些是登记数量，各实现的实际支持域与验证范围另行限定。

同一逻辑算子可以有多个后端实现；同一实现也可以提供多个公共入口。
别名、functional/out、直接函数与 wrapper 的数量不能直接当作数学算子或物理 kernel 的数量。

| 能力域 | 绑定 NPU runner 的 op 数 | 代表性公共接口 | 当前实现范围 |
| --- | --- | --- | --- |
| Attention | 8 | `single_prefill_with_kv_cache`、`single_decode_with_kv_cache`、paged/ragged batch wrapper、`single_prefill_with_kv_cache_catlass`、`sparse_flash_attention` | CANN single/batch、限定支持域的 CATLASS FA 与 Ascend910B arch22 SFA；普通 batch 路线组合 gather/FIA/concat |
| GEMM 与线性计算 | 8 | `mm_bf16`、`bmm_bf16`、`bmm_fp8`、`gemm_fp8_nt_groupwise`、`SegmentGEMMWrapper`、`matmul`、`matmul_bias_silu`、`grouped_matmul_swiglu_quant` | CANN BF16/FP8、Matmul 三后端与 Ascend C 融合算子；GroupedMatmulSwigluQuant 为 Ascend910B AOT-only |
| MoE | 3 | `fused_moe`、`fp8_per_tensor_scale_moe`、`fp8_block_scale_moe` | 基于 CANN 的路由与专家计算，受各接口契约限制；分布式 EP 数值未验证 |
| 归一化 | 5 | `rmsnorm`、`fused_add_rmsnorm`、`gemma_rmsnorm`、`gemma_fused_add_rmsnorm`、`layernorm` | CANN 路线；Gemma fused Add RMSNorm 使用 Triton |
| 激活 | 3 | `silu_and_mul`、`gelu_and_mul`、`gelu_tanh_and_mul` | 门控激活与逐元素乘法 |
| RoPE | 6 | `apply_rope`、`apply_rope_pos_ids`、`apply_rope_with_cos_sin_cache` 及各自原地版本 | 位置、频率、缓存和原地修改契约分别校验 |
| 采样与解码 | 10 | logits/probabilities 采样、top-k/top-p/min-p、重归一化、`chain_speculative_sampling` | 设备侧过滤与采样的组合实现；未声明跨平台随机数位级一致 |
| KV cache | 1 | `append_paged_kv_cache` | 普通 paged KV 追加 |
| Attention 状态合并 | 2 | `merge_state`、`merge_states` | output/LSE 合并；完整多级 cascade wrapper 尚未接齐 |

逐接口信息见 [当前登记说明](docs/operator-coverage.md#本仓当前实现全集) 与[原静态审计快照](docs/generated/operator-coverage/local-operators.md)
和 [上游接口对比](docs/operator-coverage.md)。`cudnn_batch_decode_with_kv_cache`
等兼容名称在昇腾侧使用本仓后端，不要求 CUDA/cuDNN。

### 随包预编译的 Ascend C 算子

普通 AOT wheel 会编译并携带以下五条生产算子。CANN 算子使用已安装的 TorchNPU/CANN
组件，Triton 使用自己的 JIT 路线。Ascend C 的 `op_package` 是 CANN 生成的算子交付包，
包含 host 调用/注册/tiling 库与设备二进制，由 manifest 及摘要绑定身份。

| 算子 | dtype 与支持范围 |
| --- | --- |
| CATLASS FlashAttention | FP16 或 BF16，Q/K/V/out 必须同 dtype；连续 Tensor，逻辑布局 NHD、物理格式 ND，batch=1；Q 为 `[S,H,128]`，K/V 为 `[T,H,128]`；S/T 是 128..512 范围内的 128 倍数，H=1..8；非 causal、无 mask |
| MatmulBiasSiLU | BF16、连续 ND；A 为 `[M,K]`，B 为 `[K,N]`，bias 为 `[N]`；M/N/K 是 16..256 范围内的 16 倍数；计算 `silu(A @ B + bias)` |
| Matmul | Ascend950 上 BF16、连续 ND；M/N/K 是 16..256 范围内的 16 倍数；使用本仓 AOT，缺包不转 JIT |
| GroupedMatmulSwigluQuant | Ascend910B 上 A8W8/A8W4，grouped GEMM → 反量化 → SwiGLU → token INT8；返回量化输出与 FP32 scale。完整存储/属性契约、AOT-only 构建和安装验收见 [迁移案例](docs/adding-a-custom-operator.md#61-迁移实例groupedmatmulswigluquant-走本仓-aot) |
| SparseFlashAttention | Ascend910B arch22 MLA，FP16/BF16、D=512、KV head=1；支持声明域内 BSND/TND/分页 KV，连续 ND 输入与 out；仅本仓 AOT，未接入 Triton/CANN 变体。边界与真机证据见 [移植验收](docs/reports/sparse-flash-attention-arch22.md) |

这五条 Ascend C 路径当前均为同步 eager 执行，图能力保持 `NOT_CAPTURABLE`。

## 快速开始

### 环境要求

Ascend950 的定向真机与 wheel 验证使用 Python 3.12、Torch `2.10.0+cpu`、
TorchNPU `2.10.0.post7` 和 CANN 9.2.0。其他版本组合需单独验证；其他 SoC 的探针
交叉编译结果不能作为该设备数值执行已通过的证据。
Ascend910B3 的 SFA 定向数值、AOT 与安装验证见 [arch22 验收记录](docs/reports/sparse-flash-attention-arch22.md)；
各设备记录的通过范围分别成立，不代表全量回归或其他产品已通过。

源码构建需要匹配的 Torch/TorchNPU 库与头文件、包含 `bisheng` 和打包工具的 CANN、
CMake、C++17 编译器，以及 libffi 开发头文件和库。先激活构建环境，再加载本机
CANN `set_env.sh`。CATLASS 使用 [FA 构建声明](flashinfer/ascendc/attention/single_prefill_catlass/implementation.py) 中固定的 revision。
安装完整 AOT wheel 后，执行仍需要匹配的 Torch/TorchNPU、CANN runtime 和 libffi，
无需编译器或 CATLASS 头文件。

### 编译 wheel

在已准备好的构建环境中，从仓库根目录执行：

```bash
export FLASHINFER_CATLASS_ROOT=../catlass-ref
python3 -B -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
python3 -B setup.py bdist_wheel
```

默认构建包含五条生产 AOT 算子包、构建源码输入和原生 TorchNPU 桥。
CATLASS 头文件仅在构建时使用，wheel 不携带完整 CATLASS 依赖 checkout。

编译框架也提供独立批量入口，保留按 kernel 的缓存和产物，无需为每个算子复制构建脚本：

```bash
python3 -B -m flashinfer.aot --all --cache-dir build/aot-cache --output-dir build/aot-release
```

集中登记、依赖定位、源码摘要更新及冻结 spec 的流程见 [新增算子指南](docs/adding-a-custom-operator.md)。
当前批量入口不把多个算子合并成一次 CMake 构建或同一个 vendor 包。

`FLASHINFER_AOT_ALL=0` 生成源码 wheel，不携带生产 AOT 和原生桥；执行 Ascend C 算子前须
另行准备桥及算子产物，或按策略构建缺失算子。`FLASHINFER_AOT_PREBUILT` 使用已验证算子包
和匹配的预编译桥，不调用编译器；桥通过 `FLASHINFER_ACLNN_ADAPTER_PATH` 指定，未设置时
使用源码树默认目录。缺桥或身份不符会明确失败。环境准备、其他构建模式和工具链/spec 更新
见 [wheel 与 conda 指南](README-wheel-conda.md) 和 [Torch/ACLNN 接入](docs/aclnn-torch-adapter.md)。

### 安装与验证

激活已准备好匹配 Torch/TorchNPU 与 CANN runtime 的**干净运行环境**，安装上一步生成的 wheel。
以下示例使用当前 Linux x86-64 的 wheel 文件名，从仓库根目录开始执行：

```bash
python3 -m pip install --no-deps --force-reinstall dist/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
cd tests
python3 -B flash_attention_acceptance.py
python3 -B matmul_bias_silu_acceptance.py
python3 -B -m flashinfer selftest
```

FA 脚本通过原生桥测试 FP16/BF16；独立 selftest 禁用 JIT，只运行安装包随附的 AOT 算子。
无 `--kernel-id` 时 selftest 检查整个生产登记；A2/A5 项需在匹配设备分别验收，
可按 ID 选择单项。缺包、失败或 skip 都不能记为完整通过。
脚本会检查 `flashinfer` 是否来自运行环境。验证 wheel 时应避免 editable 安装抢先导入源码，
隔离方法见 [安装说明](README-wheel-conda.md)。纯路径依赖共享不等于使用 `--system-site-packages`。

### 基本调用

安装完整 AOT wheel 后，可使用同一执行链的 Torch 入口和 Python 薄封装：

```python
import torch
import torch_npu
import flashinfer
from flashinfer.aclnn import enable_aclnn

enable_aclnn()  # 幂等注册 Torch schema，不切换执行模式。

for dtype in (torch.float16, torch.bfloat16):
    q = (torch.rand(128, 2, 128) * 2 - 1).to(dtype).npu()
    k = (torch.rand(256, 2, 128) * 2 - 1).to(dtype).npu()
    v = (torch.rand(256, 2, 128) * 2 - 1).to(dtype).npu()
    output = torch.ops.flashinfer.single_prefill_with_kv_cache_catlass(q, k, v)
    supplied_out = torch.empty_like(output)
    flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, out=supplied_out)
    print(output.shape, output.dtype)
```

Python 公共函数自动注册并转发到同名 Torch 算子；直接调用 `torch.ops` 时可在初始化阶段
显式注册一次。两种写法都必须经过原生 TorchNPU/ACLNN 桥，`enable_aclnn()` 不再切换模式。
`backend`、`kernel_id` 继续传入同一 dispatcher；Python 的 `out=` 返回原 Tensor，
Torch `_out` 原地写入后返回 `None`。缺桥、损坏或加载失败明确报错。
functional/out schema 及边界见 [Torch/ACLNN 指南](docs/aclnn-torch-adapter.md)。

### 调用方式

| 方式 | 适用接口 |
| --- | --- |
| 直接函数 | single attention、归一化、RoPE、激活、采样和普通 GEMM |
| 创建 → `plan()` → `run()` | `BatchDecodeWithPagedKVCacheWrapper`、`BatchPrefillWithPagedKVCacheWrapper`、`BatchPrefillWithRaggedKVCacheWrapper` |
| 创建 → `run()` | `SegmentGEMMWrapper`；分段元数据在 `run()` 时提供，没有公开 plan 步骤 |

batch wrapper 的 plan 保存后续运行需要的元数据和配置。兼容的新 Tensor 内容可以复用 plan；
请求结构或配置变化时需遵守对应 wrapper 的重新 plan/update 契约，具体缓存内容与失效规则
由 wrapper 定义。上述生产 FA/MatmulBiasSiLU 是直接 eager 接口，ACLNN host tiling/workspace
查询仍逐次执行。示例与公共 UT 要求见 [算子调用方式](docs/operator-call-patterns.md)。

## 当前边界

- 混合 `BatchAttention` 已有 Host/reference 生命周期覆盖，默认生产 NPU provider 尚未接通；
  普通可执行的 CANN batch wrapper 属于另一条已接入路线。
- 图 capture/replay 证据只覆盖一组受限 single-decode 配置。生产 Ascend C FA/MatmulBiasSiLU
  和普通 batch 的入图未验证；类名含 `CUDAGraph` 不会扩大验证范围。
- 上游 MLA、sparse/POD attention、完整 shared-prefix/cascade wrapper、FP4/MX 量化、通信与
  分布式算子族尚未完整实现。[差距清单](docs/operator-coverage.md) 分别记录缺失接口、
  Host 契约与可利用的后端组件。
- kernel 是否可用取决于 dtype、shape、布局、硬件和依赖。reference 执行需显式选择；
  不支持的输入、缺失的必要依赖均明确失败。当前未提供同口径性能对比结论。

## 测试与验证

按改动范围选择回归档位：

```bash
python3 -B tools/run_regression.py --profile quick
python3 -B tools/run_regression.py --profile compile tests/test_flash_attention_real.py
python3 -B tools/run_regression.py --profile full
```

`quick` 排除 `real_compile` 用例，但仍可能使用 NPU。编译相关改动补充受影响的 compile 用例；
`full` 已包含编译用例，发布验证后无需重复相同矩阵。入口保存完整日志、JUnit 和 pytest 原始
退出码，将 skip 或没有执行用例视为失败。

2026-10-29 的统一 Torch 执行验证记录为：定向 quick **181 passed**、定向 real_compile
**9 passed**，均为 **0 skipped**；隔离安装 wheel 的 selftest 为 **2 PASS、0 FAIL、0 SKIP**。
该记录包含 FA 双 dtype、MatmulBiasSiLU、AOT/冷 JIT、非默认 stream、非零 storage offset
和 TorchNPU 队列模式 0/1/2 的验证，**不是全仓完整回归结果**。
完整命令与未验证事项见 [统一 Torch 执行验收](docs/reports/torch-only-execution-2026-10-29.md)，
通用批量构建见 [编译框架验收](docs/reports/ascendc-framework-2026-10-29.md)。此前证据见
[架构重构报告](docs/reports/architecture-refactor-2026-10-28.md) 和
[FA dtype 扩展报告](docs/reports/fa-dual-dtype-2026-10-29.md)，各报告仅对应其记录的基线。

## 文档导航

| 主题 | 入口 |
| --- | --- |
| 编译、安装与直接调用算子 | [wheel/conda 快速开始](README-wheel-conda.md) |
| CANN/Triton 可运行示例 | [Ascend950 指南](docs/a5.md)、[冒烟脚本](examples/a5_smoke.py) |
| 框架架构与实现边界 | [框架设计](docs/framework_design.md) |
| 运行时与编译系统技术设计 | [数据模型、调度、AOT/JIT 与 FA 接入](docs/runtime-and-compilation-design.md) |
| 架构介绍提纲 | [项目定位与概念整理](docs/expert-architecture-overview.md) |
| FA 调用链与多后端接入 | [适配层、分发、AOT/JIT 与 ACLNN](docs/flash-attention-call-chain.md) |
| 算子全集与上游对比 | [当前登记与上游对比](docs/operator-coverage.md)、[原静态审计快照](docs/generated/operator-coverage/local-operators.md) |
| 新增 kernel 或 dtype | [算子接入指南](docs/adding-a-custom-operator.md)、[源码组织](csrc/kernels/README.md) |
| TorchNPU 队列与 ACLNN 接入 | [Torch/ACLNN 适配层](docs/aclnn-torch-adapter.md) |
| 公共 wrapper 生命周期 | [调用方式与 UT](docs/operator-call-patterns.md) |
| AOT 加载与 JIT 策略 | [产物发现](docs/aot-runtime-discovery.md) |
| Attention 详细契约 | [文档索引](docs/README.md)、[下方 Host 契约说明](#attention-框架契约) |

## 参与开发与许可证

参与开发前请阅读 [协作指南](CONTRIBUTING.md) 和 [算子接入指南](docs/adding-a-custom-operator.md)。
README 的组织参考 flashinfer-ori: `README.md`，能力描述以本仓 Ascend 实现为准。
项目使用 [Apache-2.0](LICENSE)；迁移的 CATLASS FA 实现保留其
[源码许可证](LICENSES/CANN-Open-Software-License-2.0.txt) 与集中[来源声明](THIRD_PARTY_NOTICES.md)。

## Attention 框架契约

<details>
<summary>参考开发环境、Host 框架契约与详细开发文档</summary>

### 参考开发环境

可运行的 A5（Ascend950）CANN/Triton 算子、环境配置及代码位置见
[Ascend950 开发指南](docs/a5.md)。加载 CANN 环境并配置 CATLASS 后，可使用参考 conda 配置
（Python 3.11、editable 开发安装）执行以下步骤。上文最近真机验证使用的是独立 Python 3.12
环境，参考配置不等于该版本组合已完成本轮 wheel 验证：

```bash
conda env create -f environment-a5.yml
conda activate flashinfer-a5
python examples/a5_smoke.py
```

FlashInfer 风格的 Attention 框架契约覆盖 single/batch prefill、decode、paged/ragged KV
元数据、mixed `BatchAttention`、plan/run 语义，以及用于小规模一致性用例的零依赖标量参考实现。
其 provider 与产物契约和前文可运行的算子后端分别维护。

有状态 Attention 接口由 wrapper 持有 plan：用户先调用 `plan()`，后续 `run()` 不传 plan
或 provider 句柄。provider 选择、页表物化、operation 版本绑定和调用授权均由 wrapper 内部维护。
**下面是接口契约示意，mixed BatchAttention 默认生产 NPU provider 尚未接通：**

```python
from flashinfer.attention import BatchAttention

attention = BatchAttention(kv_layout="HND", device="npu:0")
attention.plan(
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_len_arr,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    page_size,
    causal=True,
)
output, lse = attention.run(q, (k_cache, v_cache))
```

`plan()` 后可通过 `attention.plan_selection` 读取所选模式、provider operation、backend 和
registry generation 的诊断信息，其中不包含可执行 plan、module、callable 或 provider 句柄，
也不需要传给 `run()`。

框架内部通过声明式 bootstrap 组合精确 package 版本、能力证据、kernel/ABI 来源、provider
adapter 和 Tensor 物化器，形成自动 NPU resolver。当前该框架随包的集成声明为空；只有安装
经过独立验证的集成后，才会导入或调用真实 CANN/flash-attention-npu provider 包。
此处为空不影响前文普通 CANN batch wrapper 与五条生产 Ascend C 算子的执行路线。

provider 集成在进程 bootstrap 阶段原子安装。每个 NPU `BatchAttention` 实例在创建时捕获
一个不可变 resolver generation，后续集成更新不会改变已有 wrapper 的 plan/run 授权。
bootstrap 控制与面向模型的 API 分开。

量化 provider 候选还需通过额外准入：每个能力 `QuantSpec` 必须将 K/V scale，以及适用时独立的
zero-point 来源，映射到所选 catalog operation 的精确量化参数。仅有量化命名参数不构成运行支持。

paged/mixed 的 `kv_cache` 参数位可承载面向未来 provider 集成的
`AttentionOperatorQuantizedKVInput`。single prefill/decode 和 ragged prefill 保留独立 `k`、`v`
参数位，各自可携带带明确逻辑 shape 的 `AttentionOperatorQuantizedTensorInput`，再由 facade
或 wrapper 内部组合。框架解包 K/V 存储，只注入精确绑定授权的 scale、zero-point 和可选运行期
乘数；这些公共输入均不暴露 provider plan 或 callable 句柄。

文档结构与内容规则见 [文档索引](docs/README.md)，总体设计见
[框架设计](docs/framework_design.md)，Host Attention 契约架构见
[Attention 架构](docs/architecture.md)。生产 Ascend C 构建与装后执行见前文 wheel 指南。

下列文档分别定义各项 Host 框架契约：

- [Attention 框架](docs/attention/framework.md)：Attention 专用框架契约。
- [plan/run 分发设计](docs/attention/plan_run_dispatch_design.md)：wrapper 持有 plan 的生命周期，
  以及未来 CANN/flash-attention-npu provider 选择边界。
- [量化输入](docs/attention/quantization.md)：INT8/UINT8/packed-INT4 KV cache 的可执行 Host 契约。
- [正确性 trace](docs/attention/trace.md)：带版本的正确性记录与回放语义。
- [协议 trace](docs/attention/protocol_trace.md)：注入式 JIT/provider 的生命周期、stream 与资源所有权记录。
- [workspace](docs/attention/workspace.md)：容量、查询与生命周期。
- [执行身份](docs/attention/execution_identity.md)：约束复用的 plan/workspace/Tensor/capability 联合身份。
- [Tensor 契约](docs/attention/tensor_contract.md)：独立于框架的 Tensor/stride/alias/stream 适配目标。
- [Torch 元数据适配](docs/attention/torch_adapter.md)：可选的纯元数据 adapter 及显式运行验证边界。
- [数值语义](docs/attention/numerics.md) 与 [资源限制](docs/attention/resource_limits.md)：异常值和元数据准入规则。
- [能力证据](docs/attention/capability_profile.md)：带证据的后端能力声明和分发门禁。
- [分发回执](docs/attention/dispatch_receipt.md)：可重新校验的 plan/profile/kernel 选择记录。
- [kernel 产物](docs/attention/kernel_artifact.md)：结构化来源与 launcher ABI 要求。
- [launcher ABI](docs/attention/launcher_abi.md)：冻结的 Attention C/POD/error ABI、地址生成和 event lease 状态机。
- [plan 元数据格式](docs/attention/plan_metadata_wire.md)：六种模式的 CSR/页表规范线格式。
- [launch 绑定](docs/attention/launch_binding.md)：aux/run-options 契约及 TensorView/KV/aux POD 物化。
- [launch packet](docs/attention/launch_packet.md)：完整资源所有权的 Host/device lease 模型与规范 13 参数提交记录。
- [launcher/provider 生命周期](docs/attention/launcher_provider.md)：Host 产物校验、符号解析、provider 提交、完成事件和卸载。

两个 single-request 注入式 JIT 兼容入口仅建模上游 scratch、output/LSE 分配及参数顺序，
不编译或加载 Ascend 产物，其对标状态仍为 `framework`。

[Attention JIT 框架](docs/attention/jit_framework.md) 记录独立的不可变 spec、registry、显式环境、
缓存身份与特化契约。这里的注入式 Attention 路线是 Host-only，没有自行提供编译器或默认动态
加载器；它与前文生产 Ascend C 的真实 AOT/JIT 编译框架分别维护。内部选中的 `ascendc_jit`
路线必须在 wrapper `plan()` 时形成精确缓存命中绑定，再通过注入 reader 重新验证所选产物
字节、通过注入 loader 解析精确内部 module 符号，之后才能导入 package callable。
注入的 planner binder 把已加载 module 的 `plan` 入口绑定到 provider plan factory，
executor binder 再绑定同一 module、精确 package callable 与 executor。相关身份冻结到活动
plan 中，每次 `run()` 都在调用 provider executor 前重新校验。

[JSON 信封](docs/attention/json_envelope.md) 定义不可信 trace/corpus 的解码限制；
[精度验收](docs/attention/accuracy.md) 定义独立的量化漂移与未来后端误差预算，并提供可回放的
配对精度语料。[量化物理布局](docs/attention/quant_physical_layout.md) 定义显式存储、scale、
zero-point 的分块、padding、布局 catalog 与不执行转换的 converter plan。

两个 single-request API、三个 paged/ragged batch wrapper 与 mixed `BatchAttention` 共享内部
provider 选择策略，同时保留显式 Host reference 执行。公共契约和剩余能力边界见
[前端契约](docs/attention/frontend_contract.md)。

### 框架诊断

下面的回归命令运行当前 quick 档。CLI 命令用于检查或验证独立的 Host 框架契约；
case/protocol 路径是占位符，需替换为调用方的实际文件：

```bash
python3 -B tools/run_regression.py --profile quick
python3 -m flashinfer show-config
python3 -m flashinfer parity-report --scope attention
python3 -m flashinfer list-kernels
python3 -m flashinfer attention-capabilities
python3 -m flashinfer attention-replay path/to/case.json
python3 -m flashinfer attention-coverage --replay --require-complete
python3 -m flashinfer attention-corpus --pretty
python3 -m flashinfer attention-accuracy
python3 -m flashinfer attention-accuracy-corpus --pretty
python3 -m flashinfer attention-protocol-validate path/to/protocol.json
```

这些框架诊断验证契约和注入的替代实现，其成功不代表 NPU 算子数值正确或已支持入图。
quick 回归属于独立测试范围，仍可能执行 NPU 用例。

</details>

算子接口与多后端实现的目录、runner 和交付职责见[算子组织规范](docs/operator-layout.md)。
