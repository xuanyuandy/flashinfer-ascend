# T10 · plan 期的最大 workspace 查询

| | |
|---|---|
| **阶段** | P3 · 图模式 aclgraph |
| **执行序号** | 13 / 20 |
| **依赖** | T9 |
| **规模** | 小 |
| **状态** | `已完成` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [x] 完成并通过验收

---

## 阶段背景（P3 · 图模式 aclgraph）

> **路线已定**：按 aclgraph 做，torchair 最多留兼容。
> 依据：vllm-ascend 已把 aclgraph 作为默认图模式，torchair 标记废弃、v0.12.0rc1 移除、
> 最晚 2026 Q1。该仓 aclgraph 覆盖 20+ 文件，torchair 只剩一段 `ImportError` 兜底。
>
> **核心原则：本仓不捕获图，只负责让自己「可以被捕获」。** 与上游一致。

### 责任划分（动手前先读这张表）

| 动作 | 上游 GPU | 本仓昇腾 |
|---|---|---|
| 调 capture / replay | vLLM | **vllm-ascend，不下沉** |
| 决定捕获哪些 batch size、分桶 | vLLM | **vllm-ascend，不下沉** |
| 分配并持有固定 buffer | vLLM 分配后传入 | **vllm-ascend，不下沉** |
| plan/run 划分、冻结校验 | **FlashInfer** | **本仓**（照搬上游） |
| 最大 workspace 查询 | 不需要 | **本仓**（plan 期，新增） |
| 重放前改写 tiling 参数 | 不需要 | **本仓**（新增，GPU 无对应物） |

**捕获的范围是整个模型 forward**（embedding + N 层 + norm + 采样），
本仓只看得见一个算子，**结构上不可能也不应该**去捕获。

**但最后两行必须下沉**，因为它们是**算子级知识**：哪个算子、第几个参数位、
装的是什么 tiling、每步怎么变。今天它待在 vllm-ascend 的 attention 后端里，
**只是因为算子调用点在那儿**。若调用点下沉而这段不跟着下沉，
就变成「本仓发起调用、vllm-ascend 却要知道它的参数布局才能去补」——
一个算子的知识被劈在两个仓里，比现状更糟。

---

---

## 任务正文

**依赖**：T9

**为什么**：昇腾算子（`npu_fused_infer_attention_score` 等）的 **workspace 大小依赖
每步变化的 seq_len**，而重放时没有 host 参与、算不了。
解法是捕获前问一个「所有 shape 都够用」的最大值，一次分好，地址不再变。

**参考实现**：vllm-ascend 用
`torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)`
（`vllm_ascend/attention/attention_v1.py:698`、`:770` 等 9 处）。
v2 版算子对应 `_npu_fused_infer_attention_score_v2_get_max_workspace`。

**做什么**：这件事**本来就属于 `plan()`**——「提前算好、固定住」正是 plan 的职责，
落在上游已有的 plan/run 分工里，不需要引入新概念。
在 `plan()` 中按图模式开关决定是否走最大 workspace 路径。

**代价要说明**：会多占显存。需要提供开关让调用方权衡。

---

## 完成时填写

- **实际改动**：plan 期按图模式开关查最大 workspace 并接到 T9 的固定资源绑定；开关 use_max_workspace 为仅关键字参数、默认 False，需同时 use_cuda_graph=True；容量不足时在写入与发布前失败，不自动扩容
- **验收结果**：实测澄清两个 get_max_workspace 返回张量而非字节数（需 numel×element_size）；我方独立复核 seq_len ∈ {1,17,128,1024,8192,32768,131072}，plan 期最大值 37,388,288 >= 每一个，无反例，且按需用量非单调（8192/32768 高于 131072）；代价实测约按需的 1.79~2 倍；data_ptr 跨 4 次 plan + 2 次 run.out 全等；负对照 3/3；1647 passed / 3 failed
- **遗留问题**（不要顺手改，记在这里）：ragged 未公开 workspace_size；v2 仅探针无 runner；原生分页 FIA / 量化 / RoPE 路径未验证
- **提交**：`ee515e3`
