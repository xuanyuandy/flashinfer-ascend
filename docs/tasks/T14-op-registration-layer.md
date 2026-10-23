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

---

### 修订说明（2026-10-23）

**原文写的是「把 vllm-ascend 的注册层一并搬过来」，与 T13 的修订同理，该做法不成立。**

原文列举的现状经核实属实：
- `csrc/torch_binding.cpp`：**87 个 `ops.def` + 88 个 `ops.impl`**
- `csrc/torch_binding_meta.cpp`：**1,899 行** meta 实现

但有两处硬障碍：

| 障碍 | 实测 |
|---|---|
| 注册在 vLLM 自己的命名空间 | `TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)`；meta 侧 `TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ascend), Meta, ops)` |
| op 名内嵌 vLLM 语义 | `chunk_gated_delta_rule_fwd_h_vllm`、`npu_causal_conv1d_310`、`npu_recurrent_gated_delta_rule_310` 等 |
| **绑定目标不存在** | 那 88 个 `ops.impl` 绑的是 vllm-ascend `csrc/` 里的实现，而 **T13 已决定不搬那批算子**——照搬绑定层等于把 87 个 op 绑到本仓没有的实现上 |

**T8 已经建好了本仓自己的注册层**，命名空间是 `flashinfer::`
（`flashinfer::rmsnorm`、`flashinfer::fused_add_rmsnorm`），与上游 flashinfer 一致。
方向应是**扩展它**，不是并进来一个 `_C_ascend`。

---

### 做什么

1. **扩展 T8 的注册层**覆盖本仓 T13 产出的 Ascend C 算子：
   `flashinfer::` 命名空间，device 实现挂 `PrivateUse1`，meta 实现供图模式追踪。
2. **meta 实现的组织方式**可参考 vllm-ascend `torch_binding_meta.cpp` 的做法
   （1,899 行怎么组织、形状推导怎么分文件），**只学结构，不拷内容**。
3. 与 T13 的 BuildRecipe / op-package 对接：AOT 产物经 aclnn 调用时，
   注册层要能把它暴露成 `torch.ops.flashinfer.*`。
4. 保持 T8 已定的性质：真实注册需显式 `enable_torch_library()`；
   meta 只推形状/dtype、不分配设备内存；import 期不加载 torch_npu / triton。

### 不做什么

- **不搬 vllm-ascend 的 `torch_binding.cpp` / `torch_binding_meta.cpp` 内容。**
- 不引入 `_C_ascend` 命名空间。
- 算子本身的下沉是 T15/T16 的范围，本任务只做注册与 meta。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T14 <标题>`
