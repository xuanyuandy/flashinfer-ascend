# FlashInfer 调用模式、算子接口全集与 Ascend 对比

核对日期：2026-10-28。上游为 flashinfer-ori 提交 `62b5ae4e52e14d841e1b0455dc3cfa7053181741`，扫描时工作树干净；本仓为 `infer` 的 `b742c5a` 加当前架构重构工作树。硬件自检 **True 8**。本页把静态接口清单、当前实现和实际测试证据分开。

上段和 `generated/operator-coverage/` 记录原始静态审计时点，未重新扫描上游。
下文当前登记口径已按类型化作者目录更新；文件与调用链以 [组织规范](operator-layout.md)、
[作者声明](authoring-and-resources.md) 和 [接入指南](adding-a-custom-operator.md) 为准。

## 先回答调用方式

**上游不要求所有算子先创建 wrapper，再 plan/run。** 其公共入口同时包含：

| 方式 | 代表 | Ascend 应如何对标 |
| --- | --- | --- |
| 直接函数 | single prefill/decode、norm、RoPE、sampling、普通 GEMM、部分 MoE | 保持直接公共函数；不增加调用方必须操作的计划对象 |
| 直接 batch 函数 | `cudnn_batch_decode_with_kv_cache`、`trtllm_batch_context_with_kv_cache` | 由函数消费已经给出的 batch/cache 元数据，不能仅因 batch 就要求 wrapper |
| 创建 → plan → run | paged/ragged batch、mixed `BatchAttention`、sparse、MLA、cascade、POD | 构造时管理资源，plan 准备结构和调度状态，run 消费当前 Tensor；UT 按这个顺序执行并验证复用 |
| 创建 → run | `SegmentGEMMWrapper`，部分 MoE/SSM 对象 | 沿用 run-only 生命周期；Segment GEMM 没有公开 plan |
| 创建 → begin_forward → forward | shared-prefix 旧接口 | 单独登记旧式生命周期，不推断为 plan/run 的同名别名 |
| 准备函数 → 执行 | prepared decode、varlen plan 等 | 准备也可以是函数或返回对象；按照实际接口记录 |

选择方式的依据是接口是否向调用方公开需要复用的执行状态。页表、indptr、稀疏结构、调度和 workspace 常适合在 plan 中准备；同一结构的新 Tensor 可以多次 run，结构变化则重新 plan。
直接函数内部同样可能生成签名、命中编译缓存、准备 tiling/workspace 并下发 kernel。因此“直接调用”不表示绕过 Python 门面、JIT 或 ACLNN。

逐类源码证据、调用示意和本仓 UT 整改见 [调用方式与公共 UT](operator-call-patterns.md)。

## 清单范围与计数

扫描 flashinfer-ori: `tests/` 的 **1416 个 Python 文件**，得到 **15311 个静态 test 函数定义**；这不是 pytest 参数展开后的用例数，更不是在本机执行过的 CUDA 用例数。
先收集 import、别名、调用及 wrapper 绑定方法，再补充顶层和已列明公共模块的导出。有限 lazy export、继承、反射中的静态字符串和函数遮蔽子模块的情况均有专门处理及 UT。

| 清单 | 数量与含义 |
| --- | --- |
| 公共门面视图 | **1175 个函数/类导出拼写，959 个规范声明**；包含别名、准备函数、工具和配置类，不能称为 1175 个数学算子 |
| 规范声明的调用方式 | 644 个直接函数、17 个 plan/run 类、14 个 run-only 类、2 个旧式生命周期类、282 个其他类/资源；直接函数中也包含准备与工具函数 |
| 其他测试可见入口 | **4759 个函数/类/方法拼写**，包含 backend、experimental、内部 primitive、工具和生命周期方法；没有因不属于公共 facade 就丢弃 |
| 其他静态引用 | 2044 个值/常量引用、1671 个未解析属性引用；未解析不等于缺失算子 |
| 动态访问 | 88 处非字面量 getattr/hasattr，保留表达式及来源；部分名称可静态恢复，任意运行时名称不假装已经解析 |
| 本仓当前运行时登记表 | **86 条 descriptor / 48 个 op**；其中 **46 个至少绑定一个 NPU runner**，2 个仅 reference；原审计快照仍为 81/45 |

这里的“规范声明”按源码定义位置合并导出别名，仍不等于物理 kernel 数。薄封装即使调用同一 kernel，也可能有不同参数、输出、原地和状态契约；反过来一个 API 也可能选择多份 backend/特化二进制。

完整数据可按列过滤，不以一张粗粒度域表代替逐接口记录：

- [公共接口逐项对比 CSV](generated/operator-coverage/upstream-api-comparison.csv)：每个拼写、规范声明、签名、生命周期、官方候选组、本仓映射和测试引用。
- [backend 与 experimental 逐项 CSV](generated/operator-coverage/backend-and-experimental-comparison.csv)：包含下层实现、内部方法和实验接口，不能作为稳定公开承诺。
- [规范声明与别名组](generated/operator-coverage/canonical-declarations.csv)、[wrapper 生命周期及 plan/run 签名](generated/operator-coverage/wrapper-lifecycles.csv)。
- [所有测试文件及摘要](generated/operator-coverage/test-file-coverage.csv)、[动态访问](generated/operator-coverage/dynamic-lookups.csv)、[未解析引用](generated/operator-coverage/unresolved-references.csv)。
- [统计](generated/operator-coverage/summary.json)、[扫描范围与源码摘要](generated/operator-coverage/scan-manifest.json)、[官方来源及人工候选规则](operator-comparison-rules.json)。

静态分析不能执行 CUDA extension、装饰器附加的方法或任意动态导出。它们保留在原始引用和动态清单中，不能据此虚构 kernel、等价接口或执行证据。当前公共视图也不是一份“所有上游函数都具有长期兼容保证”的声明。

## 本仓当前实现全集

[原审计的 45 行快照](generated/operator-coverage/local-operators.md) 与 [CSV](generated/operator-coverage/local-operators.csv) 保留当时的入口、backend、源码行号和约束，[descriptor 快照](generated/operator-coverage/local-descriptors.json) 保留当时 81 项；它们不充当当前完整清单。当前 48 项以 `flashinfer/operators/`、各后端声明及 `tests/production_inventory.py` 的独立精确预期为准。
[本仓公开函数/类拼写](generated/operator-coverage/local-public-interfaces.csv) 另外保留别名及 Host/框架类型。CSV 中的 Torch 入口需按对应注册函数显式启用，functional/out 共享 op，不另计为新算子。qualified 名称也包含可从子模块 import 的符号；同名顶层函数可以遮蔽子模块属性，不能假定所有名称都能在一次顶层 import 后逐级 getattr。

| 能力域 | 至少绑定 NPU runner 的 op | 当前公共入口与边界 |
| --- | --- | --- |
| Attention | 8 | single prefill/decode、paged decode、paged/ragged prefill、CUDNN 兼容入口、CATLASS prefill、Ascend910B arch22 MLA SFA；不等于混合 BatchAttention、全部 MLA/sparse/POD 家族已支持 |
| GEMM | 8 | BF16 MM/BMM、FP8 BMM/groupwise、Segment GEMM、MatmulBiasSiLU、三后端 Matmul、Ascend910B GMM/SwiGLU/量化；各自受 dtype/SoC/shape 限制 |
| sampling | 10 | 概率/logits 采样、top-k/top-p/min-p、过滤/重归一化、chain speculative sampling；由设备侧过滤及采样步骤组合 |
| RoPE | 6 | 普通、pos_ids、cos/sin cache 及各自原地版本；不包含 Llama3.1 频率缩放族 |
| norm | 5 | RMSNorm、Add RMSNorm、Gemma 两种、LayerNorm；Gemma fused add 为 Triton 路径 |
| activation | 3 | SiLU/GELU/GELU-tanh 与 multiply；量化、专家打包和 backward 变体另计 |
| MoE | 3 | 非量化、FP8 per-tensor、FP8 block-scale 的有限实现；不是完整上游 MoE/EP 产品面 |
| cascade | 2 | merge_state、merge_states；没有独立 merge_state_in_place 公共入口 |
| page | 1 | 普通 paged KV append；MLA/FP4 append、公开 metadata helper 仍缺 |
| 合计 | **46** | 不把两个内部 reference-only `attention.decode` / `attention.prefill` 算作 NPU 加速 |

“NPU runner 已绑定”表示该实现可被调度，不能外推为所有 dtype、shape、量化布局、产品和图模式都通过。普通 Torch/CANN 组合也不能自动称为一个 fused kernel。此次未新增或更改任何生产算子、冻结 ABI、精确登记集合或 graph_capability。

### BatchAttention 的位置

本仓顶层确实导出了 `flashinfer.BatchAttention`，实现位于 `flashinfer/attention/holistic.py`，也有创建、plan、run 的公共 Host UT。
它通过另一套 holistic provider runtime 解析 NPU 执行；默认尚无生产 provider 闭环。普通 `BatchPrefillWithPagedKVCacheWrapper`、ragged wrapper 和 paged decode 则已有 CANN 内置执行路线。
**不能把“有 BatchAttention 类”写成“mixed batch NPU 已支持”，也不能把本仓说成完全没有这个接口。**

MoE 也有类似边界：`flashinfer/cann/fused_moe/` 已有 `ep_size>1` 的外部 `cann_ops_transformer` 接入分支；缺少依赖会明确失败。原审计未验证多 rank 执行，也没有完整的上游 `flashinfer.moe_ep` 接口族，不能简单记为“全部支持”或“源码中完全没有 EP 分支”。

## Ascend 官方能力与实现差距

下表是**候选实现路线**，不是把官方有算子写成本仓已接入。每个逐接口 CSV 通过 `candidate_group` 连接 [完整候选规则](operator-comparison-rules.json)；同一类中的具体参数和物理布局仍须单独核对。
公开文档跨 8.5、9.0、9.1 和 9.2 版本，版本及产品限制已保存在规则文件，不混用成当前 CANN 的统一支持承诺。

| 上游能力/接口族 | 官方候选与本仓差距 |
| --- | --- |
| single prefill/decode | [FusedInferAttentionScoreV4](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScoreV4.md) 同时面向 prefill/decode。本仓已有有限 CANN 路线；Q/KV layout、mask、scale、LSE 底数及输出契约仍需适配 |
| paged/ragged batch | 复用 FIA 与 [ScatterPaKvCache](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnScatterPaKvCache.md) 等组件；本仓已有公共 wrapper，但运行期仍有逐请求 gather/FIA/拼接，不能称为上游同等 fused scheduler |
| mixed BatchAttention | 可评估 FIA 加混合请求调度；本仓只有 Host/框架生命周期，缺实际 NPU provider。普通 batch wrapper 的成功不补上这个接口缺口 |
| attention sinks、POD、shared-prefix、prepared decode、DCP | FIA 与专用调度/通信可能作为组件；本仓对应完整入口、计划复用和融合路径未接齐。不能由 single attention 的通过外推 |
| block/variable sparse | [Transformer 官方目录](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/op_api_list.md) 有 sparse attention 相关能力，但 CSR/变块、稀疏索引与压缩维度不能按名字视为等价。本仓无完整 sparse 公共族 |
| MLA / sparse MLA / paged MQA logits | 官方 MLA prolog、稀疏 attention 等是候选组件；[版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md) 还有 MLA API 演进。压缩 KV、RoPE 分离、量化/cache 与 plan 需专门接入，本仓 MLA 族未实现 |
| attention 输出合并 | [AttentionUpdate](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnAttentionUpdate.md) 提供输出/LSE 合并能力。本仓两个 merge 接口可调度，in-place 和多级 cascade wrapper 仍是独立接口缺口 |
| 普通、batch、segment/group GEMM | [QuantMatmul 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnQuantMatmulV4.md) 区分普通 Mm/BatchMatMul；[GroupedMatmul 演进文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV2.md) 描述 V5 分组与量化。本仓有六个 GEMM op，segment 的计划信息由 run 参数提供 |
| FP8 / FP4 / MX / SVDQuant / LoRA GEMM | Grouped/QuantMatmul 是部分数学组件；NVIDIA scale swizzle、FP4 舍入、分组粒度及低秩融合不是通用 matmul 的别名。本仓 FP8 部分已实现，其余接口和布局未齐 |
| 单卡 fused MoE | GroupedMatmul 加 [MoeTokenPermute](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMoeTokenPermute.md)、激活、聚合可组合。本仓三接口不覆盖上游所有 backend、激活和量化模式 |
| MoE EP / allreduce / Ulysses / 通信融合 | [HCCL](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/commlib/hcclug/hcclug_000009.html) 有数学集合通信原语；CUDA IPC、NVSHMEM、MNNVL、对称内存和融合调度资源协议须重做。本仓没有完整 comm/moe_ep 公开族，多 rank 数值未验证 |
| RMSNorm / Add RMSNorm / Gemma / LayerNorm | [NN 官方目录](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/aolapi/context/ops-nn/op_api_list.md) 包含多种 norm/量化/fusion。五个本仓 API 与量化输出、gated residual、QK norm+RoPE、DiT 等额外融合分开列；基本 norm 不代表这些全部支持 |
| RoPE / Llama3.1 / QK 融合 | [RotaryPositionEmbeddingV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnRotaryPositionEmbeddingV2.md) 提供旋转组件；本仓六个 API 共享旋转能力，但频率构造、pos_ids、cache、原地协议和 QK 融合各自独立 |
| 采样、renorm、speculative chain | NN 目录中的 sampling 家族可作为组件。本仓用设备过滤/重归一化与原生分类采样组合，chain 接受/拒绝还有独立逻辑；不宣称跨平台 RNG 位级一致 |
| topk、varlen topk、page-table/index transforms | 通用 Topk/Sort 只能作为组件，变长边界、tie policy 和索引变换仍须实现；本仓缺完整公开族 |
| KV append / MLA append / metadata / packed cache | ScatterPaKvCache 可做部分更新；本仓只接普通 paged append，不能据此把 MLA、FP4 append 和 metadata kernel 一起标绿 |
| FP4/FP8/MX quantize/dequantize | NN 目录含 DynamicQuant/DynamicMxQuant 等候选；dtype 编码、scale 排布、舍入、全局/分块 scale 不同。本仓 QuantSpec 是契约，缺公开量化 kernel 族 |
| packbits / segment_packbits | 是位序/偏移的打包语义，不能用 DynamicQuant 充当等价接口；候选为位运算组合或自研，本仓未实现 |
| gated activation / 专家量化激活 | NN 激活与量化组件可作候选；本仓三种基本门控激活与量化、专家打包、backward 变体分别核对 |
| GDN recurrent / chunk / MTP | [RecurrentGatedDeltaRule 9.2 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/aolapi/context/ops-transformer/aclnnRecurrentGatedDeltaRule.md) 列出 Ascend950；[官方生态说明](https://www.hiascend.com/activities/dynamic-news/785?pageSize=10%26pageNo%3D1%26domainCode%3D%26id%3D785%26lang%3Dzh) 提及 FLA-NPU。recurrent、chunk、state pool、MTP 和 checkpoint 仍需逐一适配，本仓未接 |
| KDA | RGDR 新版的 gk 公式可作为调研入口；per-channel gate、状态方向、packed decode、prefill checkpoint 不能直接认定等价，本仓未接 |
| GDP | 不能把 gated delta product 按名字并入 gated delta rule。本轮未核实到与上游完整契约等价的公开单接口，本仓未接 |
| Mamba / causal conv / SSD / selective state update | 卷积、scan、状态更新或自研是组合方向；本轮未验证同义完整 API，本仓无对应族 |
| mHC / MSA / MiniMax / Kimi / Qwen / vision 等特化流水 | 上游测试已进入 backend/experimental 清单；matmul/norm/attention 只能作为组件，本轮没有核实完整对应实现，本仓未接 |
| logits processor / grammar | 通用过滤、比较、topk 不提供完整 processor 状态协议，本仓无对应公开族 |
| backward / autograd / 训练辅助 | [训练 attention](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnFlashAttentionScoreV2.md) 与 NN Grad 组件不等于上游训练闭环；本仓推理 forward 不能作为 backward 已支持证据 |
| autotune / trace / JIT factory / config / workspace / graph resource | 属于工程与调用协议，单独统计；不是缺一个同名 ACLNN 算子即可完成的数学能力 |

版本限制也属于 gap：例如 [RGDR 9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnRecurrentGatedDeltaRule.md) 标注不支持 Ascend950，而上述 9.2 文档支持；不能沿用旧表断言 CANN 没有相应能力。
[GemmaRmsNorm 9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnGemmaRmsNorm.md) 与 [TopKTopPSample 9.1](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnTopKTopPSample.md) 的旧产品限制，也不能替代本机 TorchNPU/CANN 组合的实际结果；本轮没有追踪这些 Python API 在该版本的完整底层 lowering。

## UT 与回归策略

接口数值 UT 已按上游生命周期整改：batch 经创建/plan/run/复用，Segment 经创建/run，直接函数继续直接调用；mixed BatchAttention 仅声明 Host/框架证据。底层 ABI/缓存/注册门禁单测保留。

日常 `quick` 运行全部非 `real_compile` 测试；`compile` 用于编译相关改动；`full` 一次已经包含 `real_compile`，不再机械地接着重跑相同矩阵。完整日志、JUnit 和 pytest 原始退出码均保存，skip 不是通过。
本轮实际结果、范围和未重跑事项见 [审计与 UT 验收](reports/operator-audit-2026-10-28.md)。

## 复现

以下复现原基线的静态审计；该 AST 工具不解析类型化 schema 动态导出的新门面，
不作为当前作者目录的生成入口。当前接入使用 `python -m flashinfer.operators generate/check`，
见 [接入指南](adding-a-custom-operator.md)。本次未重新执行上游盘点或重生成历史 CSV/JSON。

```bash
# 只解析源码；不 import 上游、不编译 CUDA、不执行上游 GPU 测试
python3 -B tools/audit_operator_surface.py --upstream ../flashinfer-ori --output-dir "$WORK/upstream"

# local-registry.json 使用原审计基线的 descriptor/能力快照；不混入当前动态导出的接口
python3 -B -m tools.build_operator_comparison --inventory "$WORK/upstream/test-api-inventory.json" --upstream ../flashinfer-ori --local-registry "$WORK/local-registry.json" --output-dir docs/generated/operator-coverage

python3 -B tools/run_regression.py --profile quick
```

原始 AST 引用、完整源码摘要、官方检索记录、设备和测试日志保存在本轮证据包中。没有执行上游 CUDA、多 GPU 通信或所有候选 CANN API；没有做性能对比。候选映射不提升本仓图能力、参数覆盖或已支持算子数量。
