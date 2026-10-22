# T3 · 把 46 条 NVIDIA 专有接口标成「有意不同」

| | |
|---|---|
| **阶段** | P1 · 接口对齐（第一步任务） |
| **执行序号** | 2 / 20 |
| **依赖** | — |
| **规模** | 极小 |
| **状态** | `自动验收通过；待用户复核` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [x] 完成并通过自动验收（用户复核待进行）

---

## 阶段背景（P1 · 接口对齐（第一步任务））

> 这是本仓的主线职责。依据是 147 条算子覆盖表与本仓登记表的三列联表。
> 当前：本仓实际接入 15 条（全表 10%）；CANN 侧「已覆盖」46 条。

---

## 任务正文

**依赖**：无。**工作量极小，但能让 T2 的范围收敛**，建议最先做。

**范围**：
- comm 域 46 条中的 NVIDIA 互联绑定：MNNVL、NVSHMEM、CUDA IPC、Lamport 协议
  （昇腾对应物是 HCCL，是另一套东西，不是「缺口」）
- `green_ctx`（CUDA SM 切分）
- `shuffle_matrix_*`（CUTLASS 布局）
- NVFP4 的 swizzle 排布相关接口

**做什么**：在对标账本（`flashinfer/data/api_parity.json` 及
`attention_api_parity.json`）中把这些条目的
`semantic_status` 设为 `intentionally_different`，并写明原因。
**不要**把它们留在「待补」列表里。

**验收**：账本中 `intentionally_different` 条目数 ≥ 46，且每条带 `reason` 字段。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T3 <标题>`

## 2026-10-21 执行结果

- 固定读取上游 `2895577720741ddfedff118202c89584996b8875`，核实并登记 59 个 NVIDIA 专属定义：CUDA IPC、vLLM CUDA AR、TRT-LLM/Lamport、MNNVL、NVSHMEM、green context 和 NVFP4 布局。逐项保存源码路径、行号、SHA-256 与 reason，不重复计算导出别名。
- Attention 原有 4 条有意差异补齐 reason；没有将可移植的 Attention 计算接口归为 CUDA 专属。
- 复现脚本：`tools/update_nvidia_inventory.py`；`ParityEntry` 拒绝没有 reason 的有意差异。
- 自动验收：`pytest -q tests/test_parity_nvidia.py`；完整结果见执行日志 `T3-verify.log`。
- 与 147 条原覆盖表的逐条对应仍归 T5；原表当前未找到，不虚构其内容。
