# flashinfer

面向推理框架集成者、算子开发者和终端用户：本仓在 FlashInfer 风格 API 下，为昇腾后端提供算子登记、路由与执行适配；本页帮助你选择可用入口并进入支持矩阵。

## 为什么使用

- 同一逻辑 op 可以绑定多个后端实现，调用方按 dtype、shape、布局、SoC 和配置选择。
- 路由结果包含候选拒绝原因、选择依据和具体 kernel ID，支持显式覆盖。
- Ascend C 的源码、能力、ABI 与构建声明进入统一生成、冻结、验证和交付链路。
- 支持范围以注册和测试定义区分；不会把 API 存在等同于目标设备全部可用。

## 本仓新增特性

- **路由主流程**：逻辑 op/Descriptor 分离，按能力准入后分栈选择；有效 tuning 优先，无记录时按可配置档位、priority 和 ID 确定选择。
- **可控覆盖和回退**：支持接口声明的 backend/kernel_id 与全局策略；fallback_chain 不做执行错误重试，仅获准的 AOT missing 进入 JIT，reference 需显式选择。
- **作者与交付入口**：公共 operator 定义、各后端 implementation/runner、可选 CMake 和数值 cases；框架生成 schema 接线、索引、摘要和冻结资源。
- **逐接口兼容**：保留直接函数、plan/run、run-only 三种形式；上游同名接口按签名与支持域核对，新增参数与不适用能力明确区分。

## 继承接口与支持范围

| 算子族 | 本仓情况 |
| --- | --- |
| Attention / KV / 状态合并 | single/batch、paged/ragged、KV append 与 merge 有受限实现；mixed BatchAttention 默认 NPU provider 不可用 |
| GEMM / MoE | BF16/FP8、分段 GEMM 和部分专家计算；本仓另有 Matmul、MatmulBiasSiLU、GMM/SwiGLU/量化直接入口 |
| Norm / Activation / RoPE / Sampling | 继承部分函数契约，提供对应后端；PDL 在 NPU 上关闭 |
| MLA / Sparse / 量化转换 / 通信 | 本仓新增 SFA 只覆盖受限 MLA；不提供上游这些完整模块与 CUDA 后端 |

## 支持摘要

| backend | 登记数 | 执行与范围 |
| --- | ---: | --- |
| `cann` | 41 | 已安装的 Torch/TorchNPU/CANN API；逐算子核对 SoC/dtype |
| `ascendc` | 5 | Ascend950 三项、Ascend910B 两项，v2 ACLNN op-package |
| `triton` | 2 | Matmul、Gemma fused Add RMSNorm，JIT |
| `reference` | 38 | 显式参考计算，不是运行失败自动兜底 |

共 48 个逻辑 op、86 条全局记录，另有独立实验实现。支持标签依据登记与测试定义，不是当前机器验收成绩；图能力和完整 dtype 组合见 [逐算子支持矩阵](docs/backends/overview.md)。

## Getting Started

使用本仓构建、与部署平台和运行栈匹配的 wheel；NPU 执行需要匹配的 Torch/TorchNPU/CANN，Triton 路线还需要 Triton Ascend。

```bash
python -m pip install ./flashinfer-0.1.0.dev0-py3-none-linux_aarch64.whl
python -m flashinfer --help
python -m flashinfer show-config
```

示例 wheel 文件名对应 aarch64；按手头产物选择。AOT-only 缺包会失败，源码模式不是其替代；构建模式与依赖见 [后端概览](docs/backends/overview.md#安装与构建边界)。

## 文档

[开始使用](docs/getting-started.md) · [架构](docs/concepts/architecture.md) · [路由](docs/concepts/backend-routing.md) · [算子支持](docs/backends/overview.md)

[接入算子](docs/development/adding-an-operator.md) · [编译交付](docs/ascendc_aot.md) · [图模式](docs/npu_graph_extensions.md) · [Tuning](docs/tuning.md) · [规划与差距](docs/roadmap.md)

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
