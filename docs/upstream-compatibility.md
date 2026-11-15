# 上游与参考实现的兼容边界

面向迁移 FlashInfer 调用的集成者：本页说明现有接口保留什么、改变什么，以及哪些族仍缺少本仓实现。

## API 基线与判定

代码台账 `tools/metadata/api_parity.json` 与 `attention_api_parity.json` 的 upstream.ref 均固定为 `2895577720741ddfedff118202c89584996b8875`。比较基于该提交的源码，不等同于当前上游主干，也不自动证明目标设备可用。

兼容分别检查名称/签名、数学语义、输入输出布局、dtype/SoC、mutation 与生命周期。静态同名和签名一致不能替代数值或执行能力；API 拼写、wrapper 方法和 Host 类型也不能算成设备 kernel 数。

## 当前接口差异

| 类别 | 例子 | 当前边界 |
| --- | --- | --- |
| 保持部分上游签名 | Activation、Sampling、single Attention、paged KV append | 仍有 dtype、SoC、参数语义和后端限制，不承诺整族等价 |
| 增加覆盖关键词 | Norm、普通 GEMM、部分 RoPE | 新增 backend/kernel_id，按本仓路由规则处理 |
| 参数位置或默认值变化 | bmm_fp8 | backend 在本仓为 keyword-only，值域和默认也不同；位置调用不能直接视作兼容 |
| 保持相应生命周期 | batch Attention 的 plan/run、SegmentGEMMWrapper 的 run-only | 不把所有接口统一改成先 plan 再 run |
| 本仓新增直接函数 | matmul、matmul_bias_silu、grouped_matmul_swiglu_quant、sparse_flash_attention | 有独立公共契约与支持域，不冒充同族上游 wrapper |
| NPU 专用扩展 | flashinfer.npu 的图参数更新和 workspace 查询 | 与上游接口分开命名，不作为 drop-in 承诺 |

`enable_pdl` 的参数位保留并在 NPU 关闭 PDL；上游 device_support_pdl 对非 CUDA 也返回 False。它只覆盖硬件启动语义，不允许丢弃 mask、scale、位置编码等计算需求。

核心 dispatcher 不接受 fa2/fa3/cudnn/cutlass 等 CUDA backend 值，不把这些名字静默映射到任意 NPU 实现。普通 single、batch facade 还各有策略解释，见 [路由](concepts/backend-routing.md)。

## 未对齐的接口面

| 方向 | 当前可用部分 | 缺失边界 |
| --- | --- | --- |
| Attention | CANN single/batch、受限 CATLASS dense FA、arch22 SFA | 完整 mixed BatchAttention NPU provider、MLA/sparse/POD wrapper 族及完整 fused 调度 |
| 低精度与量化 | 部分 FP8 GEMM/MoE、A8W8/A8W4 融合、QuantSpec | 上游完整 quantization 转换、FP4/MX 格式与全组合能力 |
| GEMM/MoE | 基础、分段、部分量化和专家组合 | 上游全部算法/后端/激活、moe_ep 与分布式产品面 |
| KV 与共享前缀 | 普通 paged append 和 output/LSE merge | MLA/FP4 追加及完整 shared-prefix/multilevel cascade wrapper |
| 通信与其他专用流水 | 没有同契约的内建公共族 | comm、Mamba、GDN/GDP/KDA、mHC/MSA 等完整模块 |
| CUDA 产物 | 无 CUDA 二进制兼容层 | CUDA module/cubin 与 CANN/Ascend C 的 ABI 不能混用 |

缺失功能不展开成“已支持”的独立使用文档。范围与补齐验收条件集中在 [规划](roadmap.md)；已登记范围见 [支持矩阵](backends/overview.md)。

## 参考项目与本仓角色

FlashInfer 提供 API 分类、函数/plan-run 生命周期及 kernel 生成组织的对照。本仓的源码/产物分层参考 `flashinfer-ori: csrc/`、`include/flashinfer/` 与 Python DSL 目录，具体执行改为昇腾后端适配。

vLLM-Ascend 提供部分原生算子的移植来源；本仓的 GMM 和 SFA 从已记录版本接入自己的 AOT、公共定义和 runner，不调用已安装的 vLLM 扩展。它们不是上游完整能力或完整 vLLM 模型集成已交付的证据。准确来源见 [集中声明](../THIRD_PARTY_NOTICES.md)。

CANN/TorchNPU 是现有算子的执行依赖，不是只凭 API 名称就可采用的能力清单。缺少本仓绑定、完整参数映射和测试的厂商/外部实现，不计为本仓支持。

## 同步方式

本仓用版本化台账保存 upstream ref、源码位置/摘要与语义状态；parity-report/parity-check 读取和校验资料。更新基线时核对实际公开符号、签名、缺省值、返回值和生命周期，再落实对应 backend/capability/tests，不用一个可解析的台账条目代替实现。

```bash
python -B -m flashinfer parity-report --scope all
python -B -m flashinfer parity-check
```

这些命令不合并上游代码，也不自动迁移新算子或证明参数等价。代码：[parity](../flashinfer/parity.py)、[CLI](../flashinfer/cli.py)、[公共门面](../flashinfer/__init__.py)。
