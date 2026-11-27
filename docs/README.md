# 文档导航

本仓在 FlashInfer 风格 API 下提供昇腾算子的登记、路由、构建和执行适配。文档以当前代码为依据，分别说明已有实现、支持限制和待实现目标。

## 按目的阅读

| 目的 | 文档 |
| --- | --- |
| 安装、检查环境并调用已有算子 | [开始使用](getting-started.md) |
| 理解分层、术语与一次调用 | [架构](concepts/architecture.md) |
| 查看选择规则、覆盖入口与失败行为 | [后端路由](concepts/backend-routing.md) |
| 查找 op × backend × dtype/SoC 的实际登记 | [后端支持矩阵](backends/overview.md) |
| 使用 Attention、CATLASS FA 与 arch22 SFA | [Attention 接口](operators/attention.md) |
| 使用稀疏索引器 Lightning Indexer | [稀疏索引器](operators/sparse-indexer.md) |
| 使用 Matmul、分段 GEMM、量化融合与 MoE | [GEMM 与 MoE](operators/gemm.md) |
| 使用 Norm、Activation、RoPE、Sampling、KV 与状态合并 | [其他 Tensor 算子](operators/tensor-ops.md) |
| 新增算子或给同一 op 增加后端 | [算子接入](development/adding-an-operator.md) |
| 理解 recipe/spec、AOT/JIT、原生桥与 wheel | [编译与交付](ascendc_aot.md) |
| 查环境变量、CLI 和配置作用域 | [配置参考](development/configuration.md) |
| 执行源码审计、工具检查与显式调优采集 | [仓库工具](../tools/README.md) |
| 提交正确性、路由与产物验证 | [测试规范](development/testing.md) |
| 维护冻结 ABI 与兼容边界 | [ABI 兼容规则](development/abi-compatibility.md) |
| 接入已有受限图能力 | [图模式](npu_graph_extensions.md) |
| 理解原生准备、图模式与 AutoTuner 执行协议 | [特性设计](features/README.md) |
| 显式采集、保存和使用调优记录 | [Tuning](tuning.md) |
| 判断上游签名、后端与参数兼容性 | [上游兼容性](upstream-compatibility.md) |
| 查看图、调优和接口扩展目标 | [规划与差距](roadmap.md) |

## 支持状态的含义

登记只说明框架知道该实现及其能力声明；测试定义只说明有可执行的覆盖入口。数值、编译、安装、图模式和性能结论需要分别对应实际运行证据。没有接通的后端不作为当前实现介绍，待补能力集中放在规划与兼容差距中。

## 事实来源

- 公共语义：`flashinfer/operators/<domain>/<op>/operator.py` 及既有领域前端。
- 后端能力和执行：`flashinfer/<backend>/<domain>/<op>/implementation.py`、`runner.py`。
- 原生计算：`csrc/kernels/<domain>/<op>/`；Triton 计算：对应后端目录的 `kernels.py`。
- 独立发布预期由 `tests/production_inventory.py` 汇总，新算子的审核记录位于 `tests/inventory/<op>.py`。
- 上游 API 基线：`tools/metadata/api_parity.json`、`attention_api_parity.json`。
- `operator-comparison-rules.json` 是静态对比工具的机器输入，候选映射不构成本仓支持声明。

历史任务报告、临时测试输出、旧扫描快照和演示附件不作为正式文档维护。生成 JSON 与本地测量留在构建/运行目录；正式文档不使用内部任务号或优先级编号。
