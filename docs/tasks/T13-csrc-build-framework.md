# T13 · 建立本仓自己的 Ascend C 编译链路（catlass 作为 3rdparty）

| | |
|---|---|
| **阶段** | P4 · 自定义算子与下沉 |
| **执行序号** | 16 / 20 |
| **依赖** | T6.1 |
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

**依赖**：T6.1（登记表能装编译产物）—— **顺序不能反**，
否则编出来的东西无处登记、选择器也选不到。

---

### 修订说明（2026-10-23）

**本任务原先写的是「整体搬运 vllm-ascend 的 csrc/」，该前提经核实不成立，已重写。**

原文的论据是「`csrc/` 本来就是一个自包含的 CANN 算子工程，不依赖 vLLM 任何概念」。
实测：**1,179 个 cpp/h 里有 117 个引用 vllm**，且是结构性耦合而非版权头——

| 耦合形式 | 实测 |
|---|---|
| `namespace vllm_ascend { }` | 67 处 |
| 类型名内嵌 Vllm | `ChunkGatedDeltaRuleFwdHVllmTilingData`、`KvQuantSparseFlashAttentionVllmTilingDataMla` |
| 导出的 aclnn 符号名 | `aclnnVllmAscendQuantLightningIndexerV2Metadata...` |
| Python 耦合 | `from vllm_ascend.utils import enable_custom_op` |
| 构建开关耦合 | `#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` |

整体拷入会把 `vllm_ascend` 命名空间、`*Vllm*` 的 tiling 结构体与 `aclnnVllmAscend*`
导出符号带进一个**以对标 flashinfer 为目的**的仓，下游拿本仓当 flashinfer 替身用时直接撞上。

**而且会与 T15/T16 冲突**：T15 是「下沉通用计算算子」、T16 是「**剥离 vLLM 语义后**
再下沉融合算子」。原计划本就是挑选 + 剥离语义地下沉；整体拷贝等于提前用错误方式做掉，
且做完后 T16 的「剥离语义」再也无从下手。

---

### 对标上游确定的结构

```
上游 flashinfer:  自己的 csrc/（3,065 个文件） + 3rdparty/cutlass（submodule）
本仓:             自己的 csrc/                + 3rdparty/catlass（submodule）
```

**catlass 是 CANN 侧的 CUTLASS。** 依赖方向应为
`vllm-ascend → flashinfer-ascend → catlass / CANN`，照搬 vllm-ascend 的 csrc 会把方向拧反。

catlass 在 vllm-ascend csrc 中仅 **9 个文件、2 个子目录**（attention、mc2）用到，
属轻依赖，适合 submodule，无需 vendor。

- 仓库：https://gitcode.com/cann/catlass
- 参考（**只读参考，不作搬运源**）：https://github.com/vllm-project/vllm-ascend

---

### 已查清的事实（T13A 探路所得）

- 本机 CANN Ascend C 工具链**可用于 ascend950（A5）**：标准 op-package 构建实际产出
  `packages/vendors/<vendor>/op_impl/ai_core/tbe/kernel/ascend950/<op>/*.o`
  与 `kernel/config/ascend950/binary_info_config.json`。
- T18B 已确认：`bisheng` / `ccec` 存在（clang 15.0.5），**无 `ascendc` CLI**；
  最小设备源码可编译为 NPU relocatable ELF；**host wrapper 与 launch ABI 尚未集成**，
  故 Ascend C 编译钩子仍显式抛 `NotImplementedError`。本任务要把它变成真能跑。

---

### 做什么

1. **catlass 接入**：`3rdparty/catlass` 作为 git submodule
   （commit `41bf90da655bba3c66d0acd7e00abe33960ecfd6`），对标上游 3rdparty 布局。
2. **本仓自己的 `csrc/`**：建立目录与构建集成（CMake / `setup.py` 的
   `CMakeExtension` / `cmake_build_ext`、SOC 维度）。vllm-ascend 的 `csrc/build.sh`
   与 `CMakeLists.txt` 可作为**做法参考**，但按本仓布局与命名重写，
   不保留 `vllm_ascend` 命名空间、`*Vllm*` 命名或 `VLLM_*` 开关。
3. **打通编译钩子**：把 Ascend C 的 `NotImplementedError` 变成真实实现，
   复用 18A/18B 的 JitBuildStore（锁 / staging / 原子发布），不另起缓存。
4. **按 T6.1 字段登记**编译产物（`ArtifactRef` / `SourceRef` / `target_soc`）。
5. **AOT 与 JIT 两条路径**：发版前按 SOC 出包；用到时现编。

**SOC 维度**：`ascend910b1`（A2）/ `ascend910_9391`（A3）/ `ascend310p1` / `ascend950*`（A5）。

### 不做什么

- **不搬 vllm-ascend 的 1,179 个算子源码。** 算子下沉是 T15/T16 的范围，
  且必须挑选 + 剥离 vLLM 语义。
- 不 vendor catlass。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T13 <标题>`
