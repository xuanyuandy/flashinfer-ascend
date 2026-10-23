# T9 · 把图契约接到 `plan()` 上（上游那一半）

| | |
|---|---|
| **阶段** | P3 · 图模式 aclgraph |
| **执行序号** | 12 / 20 |
| **依赖** | T1 |
| **规模** | 中 |
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

## ⚠ 与 T4.4 的分界

两条任务都动 decode wrapper，**分工必须守住**：

| | 负责 |
|---|---|
| **T4.4**（P1，先做） | 公开 API 形状、分页参数（`indptr` / `indices` / `last_page_len`）、**eager 行为** |
| **T9（本任务）** | 固定 buffer、**图生命周期**（冻结语义、捕获后不可变） |

**前提**：进入本任务时 T4.4 应当已经完成——
`plan()` 的参数集已经对齐上游、eager 路径已经正确。
**本任务只在它之上叠加图相关的部分**，不要重新调整公开 API 形状。

---

## 任务正文

**依赖**：T1

**文件**：`flashinfer/attention/execution_identity.py`、
`flashinfer/attention/batch.py`、`flashinfer/decode.py`、`flashinfer/prefill.py`

**好消息：方向已经对了，形状也已经有了。**
`execution_identity.py:95` 的 `AttentionGraphResourceContract` 已声明
`graph_enabled` / `fixed_batch_size` / `persistent_buffers`，
并校验「开了图就必须有固定 batch、固定 buffer，且都在同一设备上」。
这是上游那一半的忠实移植。

**缺的是把它真正接到 `plan()` 上。**

**做什么**（逐条对标上游 `flashinfer-ori/flashinfer/decode.py`）：
1. wrapper **构造时**收固定 buffer（上游是 `paged_kv_indptr_buffer` /
   `paged_kv_indices_buffer` / `paged_kv_last_page_len_buffer` 三个）。
   本仓该构造函数目前只有 2 个参数，上游有 9 个。
2. `plan()` 只往固定 buffer 里 `copy_`，**不重新分配**，地址永不变。
3. **冻结语义**：batch size、后端选择、`q_len_per_req` 一旦首次捕获就冻住，
   下次 `plan()` 换了值**直接报错**，不要悄悄换 kernel——那会让已录的图对不上。
   上游的冻结检查覆盖 batch size、`q_len_per_req`、`backend="auto"` 的解析结果、
   block table 宽度，**全部「变了就抛错」**。
4. 删掉 `flashinfer/decode.py:158` 那句误导性注释：
   ```python
   del workspace_buffer, is_cuda_graph_compatible  # NPU 无对应机制，接受并忽略
   ```
   **aclgraph 就是那个对应机制。** 这句话会让后面接手的人以为这件事不需要做。

**命名**：对外参数名建议保留上游的 `use_cuda_graph` / `is_cuda_graph_compatible`
以维持 drop-in（内部是 backend-neutral 的 graph 标志，
本仓 `attention/batch.py:234` 已有 `is_cuda_graph_enabled` 作为兼容别名，沿用这个模式）。

**验收**：构造 wrapper 时传固定 buffer → 两次 `plan()` 用不同 batch size → 第二次抛错。

---

## 完成时填写

- **实际改动**：plan() 对真实 NPU 固定 buffer 原地 copy_；补齐 resolved_backend 与 block_table_width 两个冻结维度（原已有 batch、q_len_per_req）；拒绝同后端悄悄换 kernel；清除 decode.py/prefill.py 两处「NPU 无对应机制」误导描述
- **验收结果**：三种 wrapper 的固定 buffer 两次 plan() 前后 data_ptr 完全相同；四维冻结均抛 AttentionStateError 并带前后值；block table 宽度按要求比上游更严（增长/缩小/有无变化一律拒绝）；我方源码级负对照：移除保护后 10 failed，还原 21 passed；1613 passed / 3 failed；79/79 选型不变
- **遗留问题**（不要顺手改，记在这里）：native run 仍有 KV 整理与临时分配，未宣称完整重放能力；外部 provider 图绑定仍显式报未实现
- **提交**：`3d8a6d2`
