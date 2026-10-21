# T11 · 通用的「重放前参数更新」机制 ⭐ 本阶段核心

| | |
|---|---|
| **阶段** | P3 · 图模式 aclgraph |
| **执行序号** | 14 / 20 |
| **依赖** | T6.5 T10 |
| **规模** | 大 |
| **状态** | `未开始` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] 完成并通过验收

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

**依赖**：T6.5（入图能力字段）、T10

**这是本仓相对上游要多承担的那一层，也是从 vllm-ascend 接手的核心资产。**

**vllm-ascend 现状**（要接手的东西）：
- `update_graph_params` **在 8 个 attention 后端里各写了一份**：
  `mla_v1` / `attention_v1` / `sfa_v1` / `dsa_v1` / `dsa_v41` /
  `mla_cp` / `dsa_cp` / `msa_m3`（共 9 处定义）
- 每份都手工把十几个参数**按位置解包、改掉、再塞回去**
- 两处带着作者自己的注释：
  ```
  # vllm_ascend/attention/mla_v1.py:954
  # FIXME: Behold! We are using a temporary hack here to update the args
  #        for each layer's attention op in the graph.
  ```

**有利的时间点**：vllm-ascend **自己也在收敛这件事**——
新写了通用的 `UpdatableGraph`（`vllm_ascend/compilation/updatable_graph.py`，217 行），
把「记句柄 → 重放前改参数」抽成框架能力，已有 12 个文件在用，
`attention_v1` 甚至把自己的手写版改成了「请走 UpdatableGraph」的报错。
**但迁移只做了一半**（`mla_v1` / `mla_cp` 仍在直接手写）。
**现在接手比等他们迁完再搬更省事。**

**底层 API**（CANN / torch_npu 公开能力）：
- `torch.npu.graph_task_update_begin(update_stream, handle)` / `graph_task_update_end()`
  —— 把新参数写回已录制的任务
- `torch.npu.ExternalEvent` —— 保证「改完再放」的顺序
- `torch_npu._C._NPUTaskGroupHandle` —— 捕获时记下的任务句柄

**做什么**：
1. 在 runtime 层实现一个通用机制：捕获期记录任务句柄与参数布局，
   重放前按 `graph_capability` 字段统一执行更新。
2. 由 `KernelDescriptor` 自己交代「要改哪些参数、怎么改」，
   **不要让每个算子各写一份**。
3. 对外暴露给 vllm-ascend 调用的接口：
   - 查询某个 kernel 的入图性质（取代 `issubclass` 判断）
   - 在 replay 前执行更新的回调

**注意：这会让本仓多出上游没有的 API 面。**
T10、T11 两项在上游**根本不存在对应接口**。
处理方式：放进明确的 NPU 扩展命名空间，**在对标账本里登记为「有意不同 / NPU-only 扩展」**，
不要让它看起来像上游 drop-in 接口的一部分，否则下游拿本仓当上游替身用时会撞上差异。

**验收**：一个 FULL 模式 aclgraph 捕获 + 多次不同 seq_len 的重放，结果与 eager 一致。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T11 <标题>`
