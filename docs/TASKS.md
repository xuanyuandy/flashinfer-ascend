# flashinfer-npu 对标与演进任务清单

> 这份清单是给执行 agent 用的工作单。每条任务都写了**依赖、改哪里、验收标准**。
> 架构背景与论证见配套的架构说明文档；这里只写「做什么」。
>
> 基线：本仓 `a65a89e` · 上游 `flashinfer-ori @ 289557772` · vllm-ascend 本地检出
> 最后更新：2026-10-21

> **当前状态：本轮仅更新规划，T1 尚未开始。**
> 下一步交付物是 **T1 的「符号与字段对照表」**，供确认最小修改范围后再动手。

---

## 怎么用这份清单

1. **按阶段顺序执行。** P0 不完成，P1 之后的任务大部分无法验证（七成模块打不开）。
2. **同一阶段内的任务可并行**，除非任务里写了 `依赖：`。
3. **每条任务独立可验收。** 验收标准是可执行的命令或可检查的事实，不是「看起来对了」。
4. **不要扩大范围。** 如果执行中发现相邻问题，记录下来，不要顺手改——这份清单的顺序是按依赖排的。

### 交付节奏（每条任务都走这一轮）

```
修改  →  自测，交付 diff / 执行的命令 / 实际结果  →  等待验证通过  →  提交  →  下一条
```

**没有通过验证之前不要提交，也不要开始下一条。**

### 验收粒度

- **T2 / T4 / T6 按子项逐个验收**（T2.1…T2.6、T4.1…T4.6、T6.1…T6.5）。
- **父任务的索引勾选框，要等它全部子项都通过之后才打勾。**
- 其余任务整条验收。

### 全局约束

- **对标优先于优化。** 任何与上游行为不一致的地方，要么改成一致，要么在对标账本里显式登记为「有意不同」并说明原因。不允许存在未登记的静默差异。
- **不可实现的东西要明确报错，不要静默降级成不同的结果。** 例外：上游自己就是静默降级的行为（见 T4.6）。
- **新增接口必须同时更新对标账本**（`flashinfer/data/*.json`）。
- 提交信息用中文，一条任务一个提交，标题带任务号，例如 `T2.1 接入 sampling 的 10 个接口`。

---

---

## 执行顺序与阶段目标

| 阶段 | 任务 | 目标 |
|---|---|---|
| **P0** | T1 | 恢复导入与测试收集，形成真实失败清单 |
| **P1** | T3 → T2 → T4 → T5 | 标明平台差异、补接口、修语义、完善账本 |
| **P2** | T6 → T7 | 扩展登记表并调整调度规则 |
| **P2+** | T18 → T19 → T20 | 打通编译加载、两级选择和性能数据 |
| **P3** | T8 → T9 → T10 → T11 → T12 | 注册/meta、图契约、workspace、参数更新 |
| **P4** | T13 → T14 → T15 → T16 | 建本仓编译链路与注册层，再挑选并剥离语义地下沉算子 |
| **P5** | T17 | 把 Attention 收敛到统一架构 |

---

## 实施前要划清的三个边界

> 这三处是**范围归属问题**，不划清会出现「两条任务改同一处」或「范围悄悄膨胀」。

### 边界一 · T1 的实际范围大于列出的 10 个符号

列出的 10 个符号是**当前能看见的**。实际依赖更多，原因是
**import 在第一个缺失符号处就中断了，后面的还没机会报出来**——
现在按首个错误归类只能看到 5 类：

| 首个报错 | 阻塞模块数 |
|---|---|
| `SchemaError` | 66 |
| `ArtifactRef` | 2 |
| `ArtifactFormat` | 2 |
| `QuantSpec` | 1 |
| `triton` 未安装（**与 schema 无关**） | 1 |

补上 `SchemaError` 之后，被它挡住的 `ArtifactKind` / `KernelBinaryABI` /
`KernelLaunchABI` / `KernelConstraints` / `KernelCapabilityBinding` /
`WorkspaceFormula` 才会浮出来。

**CLI 与入口同样被阻塞**（不只是库内部）：

```
python -m flashinfer --help   →  ImportError: cannot import name 'SchemaError'
import flashinfer.cli         →  同上
import flashinfer.parity      →  同上（从 .runtime.schema 导入）
```

注意两条不同的导入路径都要满足：`flashinfer.runtime.schema` 和
`flashinfer.runtime`（即 `schema.py` 定义 + `__init__.py` 再导出）。

**做法**：先整理**完整的契约对照表**（每个符号的使用点 → 反推出的字段/签名），
**确认最小修改范围后再动手**，不要边补边猜。

### 边界二 · T2.2 与 T4 有重叠，先定归属

T2.2（Attention 后端接入）与 T4（现有接口纠偏）会碰同一批 attention 接口。
**进入 T2.2 之前先明确每一处改动归哪条任务**，避免同一处被改两次或都没改。

### 边界三 · T4.4 与 T9 的分界

两条都动 decode wrapper，分工如下：

| | 负责 |
|---|---|
| **T4.4** | 公开 API 形状、分页参数（`indptr` / `indices` / `last_page_len`）、**eager 行为** |
| **T9** | 固定 buffer、**图生命周期**（冻结语义、捕获后不可变） |

**T4.4 先做**（P1），T9 在它之上加图相关部分（P3）。

---

## 任务索引

20 条任务，**按执行顺序排列**。每条点进去是一份自带依赖、文件清单、验收命令的工作单。

> 勾选框就是进度。改完一条，在这里和任务文件里同时打勾。

### P0 · 解除阻塞

> 这一阶段不做，后面所有任务都无法验证。全仓 104 个 Python 模块里 72 个今天 import 就失败。

- [ ] **[T1](tasks/T01-runtime-schema-symbols.md)** · 补齐 10 个缺失的 runtime 基础定义  
  <sub>依赖 — · 规模 大 · 执行序号 1 · 导入/收集门禁通过，ABI 指纹待修复；[报告](reports/T1-report.md)</sub>

### P1 · 接口对齐（第一步任务）

> 这是本仓的主线职责。依据是 147 条算子覆盖表与本仓登记表的三列联表。
> 当前：本仓实际接入 15 条（全表 10%）；CANN 侧「已覆盖」46 条。

- [x] **[T3](tasks/T03-mark-nvidia-only.md)** · 把 46 条 NVIDIA 专有接口标成「有意不同」  
  <sub>依赖 — · 规模 极小 · 执行序号 2 · 已核实 59 个 NVIDIA 专属定义，自动验收通过</sub>
- [ ] **[T2](tasks/T02-cann-interface-backfill.md)** · 补齐 32 条「CANN 有、本仓没接」的接口  
  <sub>依赖 —（验证需 T1） · 规模 大 · 执行序号 3 · T2.1/2.2/2.5 自动验收通过；T2.6 已接入 cascade/page；其余待覆盖原表</sub>
- [x] **[T4](tasks/T04-silent-parity-defects.md)** · 修掉 6 处会静默出错的对标缺陷
  <sub>依赖 — · 规模 中 · 执行序号 4 · 六项完成，parity/enum/pdl/backend 门禁 131 项通过；用户尚未复核</sub>
- [ ] **[T5](tasks/T05-parity-ledger.md)** · 让对标账本可复现、可运行、覆盖完整  
  <sub>依赖 T1 · 规模 中 · 执行序号 5 · 双账本固定提交、校验与 CI 已接通；原始 147 条覆盖表待提供，保留 bootstrap</sub>

### P2 · 地基扩展

> **四件事里有三件卡在同一个地方**：登记表能描述的东西太少。
> 这不是某个功能的缺陷，是公共地基的缺陷。

- [x] **[T6](tasks/T06-registry-fields.md)** · 给登记表加 5 类字段  
  <sub>依赖 T1 · 规模 大 · 执行序号 6</sub>
- [x] **[T7](tasks/T07-dispatch-ordering.md)** · 重做 dispatch 排序规则  
  <sub>依赖 T6.1 T6.2 · 规模 小 · 执行序号 7</sub>

### P2+ · 选型与编译的分层

> 这三条补的是「JIT 编译一整套」和「选取性能最优实现」这两个诉求里
> **前面阶段没覆盖到**的部分。T6 加的是**字段**，这里加的是**分层结构**。

- [x] **[T18](tasks/T18-jit-trigger-layer.md)** · 统一的多后端 JIT 触发层  
  <sub>依赖 T1 T6.1 · 规模 大 · 执行序号 8</sub>
- [x] **[T19](tasks/T19-two-level-dispatch.md)** · 把 dispatch 拆成两级  
  <sub>依赖 T6.3 T6.4 T18 · 规模 大 · 执行序号 9</sub>
- [x] **[T20](tasks/T20-perf-data-pipeline.md)** · 性能数据链路  
  <sub>依赖 T19 · 规模 大 · 执行序号 10</sub>

### 计划外完成（执行中发现的真实缺陷与工程改进）

> 这些不在原清单里，是执行 T6~T20 时发现并单独立项修掉的。
> 每条都有可复现的证据，提交信息里记了复现方式。

- [x] **修复 `KernelRegistry.select()` 与 dispatch 的策略执行不一致** · `fd94304`
  <sub>2 处真实绕过：自定义 kernel 的 opt-in 被defeat；capability（dtype/layout/SoC/核数/特化参数）在 select() 侧完全没检查，910B-only kernel 可在 Ascend950 上被选中</sub>
- [x] **测试按「是否真实调用编译器」分组（`real_compile`）** · `0bf3e90`
  <sub>依据审计钩子记录的真实编译器调用，非文件名或耗时；默认行为不变</sub>
- [x] **修复决策缓存跨 registry 串用** · `dd73c9b`
  <sub>缓存 key 缺 registry 身份，新建 registry 的 revision 同为 1 时撞 key，registry B 会返回 B 中不存在的 kernel。既有缺陷，在基线提交上同样复现</sub>

### P3 · 图模式 aclgraph

> **路线已定**：按 aclgraph 做，torchair 最多留兼容。
> 依据：vllm-ascend 已把 aclgraph 作为默认图模式，torchair 标记废弃、v0.12.0rc1 移除、
> 最晚 2026 Q1。该仓 aclgraph 覆盖 20+ 文件，torchair 只剩一段 `ImportError` 兜底。
>
> **核心原则：本仓不捕获图，只负责让自己「可以被捕获」。** 与上游一致。

- [x] **[T8](tasks/T08-torch-library-meta.md)** · 建立 torch.library 注册层与 meta 实现  
  <sub>依赖 T1 · 规模 中 · 执行序号 11</sub>
- [x] **[T9](tasks/T09-graph-contract-plan.md)** · 把图契约接到 plan() 上  
  <sub>依赖 T1 · 规模 中 · 执行序号 12</sub>
- [x] **[T10](tasks/T10-max-workspace.md)** · plan 期的最大 workspace 查询  
  <sub>依赖 T9 · 规模 小 · 执行序号 13</sub>
- [x] **[T11](tasks/T11-replay-param-update.md)** · 通用的「重放前参数更新」机制 ★  
  <sub>依赖 T6.5 T10 · 规模 大 · 执行序号 14</sub>
- [x] **[T12](tasks/T12-rewrite-design-3-4.md)** · 重写设计文档 §3.4  
  <sub>依赖 T9 T10 T11 · 规模 小 · 执行序号 15</sub>

### P4 · 自定义算子与下沉

> **终态**：vllm-ascend 从「自己持有 66 个 CANN 算子调用点 + 一套算子工程 + 8 份入图补丁」
> 变成**只调本仓接口**；本仓从「对标用的壳」变成**真正持有实现的那一层**。

- [ ] **[T13](tasks/T13-csrc-build-framework.md)** · 建立本仓自己的 Ascend C 编译链路（catlass 作为 3rdparty）  
  <sub>依赖 T6.1 · 规模 大 · 执行序号 16</sub>
- [ ] **[T14](tasks/T14-op-registration-layer.md)** · 搬运算子注册层  
  <sub>依赖 T8 T13 · 规模 中 · 执行序号 17</sub>
- [ ] **[T15](tasks/T15-sink-generic-ops.md)** · 下沉通用计算算子  
  <sub>依赖 T13 T14 · 规模 大 · 执行序号 18</sub>
- [ ] **[T16](tasks/T16-sink-fused-ops.md)** · 剥离 vLLM 语义后再下沉融合算子  
  <sub>依赖 T15 · 规模 大 · 执行序号 19</sub>

### P5 · 收尾

- [ ] **[T17](tasks/T17-attention-dir-disposition.md)** · 处置 attention/ 那两万多行  
  <sub>依赖 T1 T6.2 · 规模 大 · 执行序号 20</sub>

---

## 三条最容易搞反的顺序

1. **登记表字段（T6.1）要在接编译器（T13 / T18）之前。**
   哪怕明天就把 Ascend C 编译器接上，编出来的东西也无处登记——
   执行体字段现在只能放一个 Python 函数。

2. **接口对齐（T2）要在下沉（T15 / T16）之前。**
   vllm-ascend 要调的那些接口，现在本仓大部分还不存在。没有接口就谈不上下沉——
   **这两件事不是并行的两条线，是同一条线的前后两段。**

3. **入图能力字段（T6.5）要和登记表改造一起做**，不要单独开一轮图模式改造。

**另外：规模区间字段（T6.3）一条挡三条**——两级选择的第一级要用它筛选、
性能数据要用它做索引、图模式按 bucket 固化选择也要用它。在 T6 内部优先级要靠前。

---

## 三项诉求的覆盖对照

| 诉求 | 覆盖 | 对应任务 |
|---|---|---|
| JIT 编译一整套（多后端触发、kernel↔符号映射） | 部分 → 已补 | T6.1 · T13 · T14 · **T18** |
| (dtype/shape/capability) → 实现栈 → 栈内配置 | 原先没有 → 已补 | T6.3 · T6.4 · **T19** |
| 选取性能最优 kernel 实现 | 原先没有 → 已补 | T7 · **T20** |
| aclgraph 整套支持 | 覆盖 | T8 ~ T12 |
| 客户自定义算子 + 接口注册 | 覆盖 | T6.1 · T6.2 · T13 · T14 |

**aclgraph 的边界**：本仓覆盖的是**算子侧的可捕获契约**（plan/run 冻结语义、
最大 workspace、重放前参数更新）。FULL/PIECEWISE 调度策略、捕获哪些 batch size、
多卡通信算子入图、投机解码场景——**这些在 vllm-ascend 侧，不下沉**。

---

## 附：待核实项

以下几条在规划阶段未能逐条验证，**实现前先查 CANN 算子清单确认**：

| 项 | 要确认什么 |
|---|---|
| `npu_gemma_rms_norm` | 是否确实没有 Add 融合版本（本仓目前用 Triton 自研补的） |
| `top_k_top_p_sample_v2` | 是否支持 MinP（影响 T2.1 的 `min_p_sampling_from_probs`） |
| `npu_scatter_pa_kv_cache` | 确切签名（影响 T2.6 的 `append_paged_kv_cache`） |
| `mega_moe` | 量化路径覆盖范围 |
| `moe_gating_top_k` | 覆盖表里写这个名字，检索到的是 `npu_moe_gating_top_k_softmax`，接入时对确切签名 |

## 附：一个已经走通的完整案例

`norm.gemma_fused_add_rmsnorm` 是**全仓唯一**走完整条路的实例：

```
对标表发现缺口 → 自研算子补上 → 登记进表 → 选择器自动选中
   （对标）      （自定义算子）   （注册开放）    （地基）
```

CANN 没有 Add+Gemma 的融合版本，本仓用 Triton 写了一个补上并登记进表。
做法上还有一个细节值得沿用：**登记时不 import triton，真正执行才加载**，
所以没装 triton 的机器照样能启动和做选择。

**架构本身是通的，后面就是把这条路重复 N 遍。**
