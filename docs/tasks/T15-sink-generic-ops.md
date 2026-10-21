# T15 · 下沉通用计算算子

| | |
|---|---|
| **阶段** | P4 · 自定义算子与下沉 |
| **执行序号** | 18 / 20 |
| **依赖** | T13 T14 |
| **规模** | 大 |
| **状态** | `未开始` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] 完成并通过验收

---

## 阶段背景（P4 · 自定义算子与下沉）

> **终态**：vllm-ascend 从「自己持有 66 个 CANN 算子调用点 + 一套算子工程 + 8 份入图补丁」
> 变成**只调本仓接口**；本仓从「对标用的壳」变成**真正持有实现的那一层**。

---

## 任务正文

**依赖**：T13、T14

**范围（这类没有疑问，先做）**：进出就是张量、与推理框架无关的算子。
- `grouped_matmul_swiglu_quant`（及 `_v2` / `_weight_nz_tensor_list` 变体）
- `dequant_swiglu_quant` / `swiglu_group_quant`
- `add_rms_norm_bias` / `rms_norm_cast`
- `bgmv_shrink` / `bgmv_expand` / `sgmv_shrink` / `sgmv_expand`（LoRA）
- `dequant_situ_quant` / `situ_mx_quant` / `gmm_dequant_situ_quant`

**做法**：下沉一个，在本仓登记一个，vllm-ascend 侧改成调本仓接口，删掉自己那份。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T15 <标题>`
