# T14 · 搬运算子注册层（torch.library + meta）

| | |
|---|---|
| **阶段** | P4 · 自定义算子与下沉 |
| **执行序号** | 17 / 20 |
| **依赖** | T8 T13 |
| **规模** | 中 |
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

**依赖**：T8、T13

**vllm-ascend 现状**：
- `csrc/torch_binding.cpp`：**87 个 `ops.def` + 88 个 `ops.impl`**，
  注册在 `TORCH_LIBRARY_EXPAND(_C_ascend, ops)`，实现挂 `torch::kPrivateUse1`
- `csrc/torch_binding_meta.cpp`：**1,899 行** meta 实现（图模式追踪必需）
- Python 侧：`vllm_ascend/ops/register_custom_ops.py` 等处的
  `direct_register_custom_op` / `torch.library.register_fake`

**做什么**：把这层一并搬过来，与 T8 建立的注册层合并。
这层决定下沉后两边怎么对接。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T14 <标题>`
