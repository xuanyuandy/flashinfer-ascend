# T13 · 把 vllm-ascend 的 `csrc/` 编译框架搬过来

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

**为什么能搬**：`csrc/` 本来就是一个**自包含的 CANN 算子工程**，不依赖 vLLM 任何概念——
输入是 Ascend C 源码加一个 SOC 型号，输出是算子包和一个 `.so`。
整体搬过来，vllm-ascend 侧不会有业务逻辑受影响。

**为什么该搬**：本仓今天**连一行 Ascend C 源码、一条编译链路都没有**
（`kernels/ascendc` 是空目录，全仓 `subprocess` 0 处、`ctypes` 0 处），
而那边是一套已经在生产里跑的完整实现。**这不是「多接一个仓」，是把缺的地基直接拿过来。**

**要搬的东西**：

| 内容 | 规模 |
|---|---|
| Ascend C 算子源码 | `csrc/` 下 **1,179 个 cpp/h**，标准 CANN 工程布局（`op_host` / `op_kernel` 成对） |
| 构建脚本 | `csrc/build.sh` —— 按 `SOC_VERSION` 产出 `ophost / opapi / opgraph / onnxplugin` 四类目标；含 `ENABLE_BUILT_JIT` / `ENABLE_BUILT_CUSTOM` 开关 |
| aclnn 构建 | `csrc/build_aclnn.sh` |
| CMake 集成 | `csrc/CMakeLists.txt`、`setup.py` 的 `CMakeExtension` / `cmake_build_ext` |
| 第三方依赖 | catlass（git submodule） |

**SOC 型号维度**（`setup.py` 里的实际取值，T6.1 的字段要能表达这些）：
`ascend910b1`（A2）/ `ascend910_9391`（A3）/ `ascend310p1` / `ascend950*`（A5）

**做什么**：
1. 搬运目录结构与构建脚本，接进本仓的 `setup.py`。
2. 让编译产物能按 T6.1 的字段登记进登记表。
3. 打通 AOT（发版前按 SOC 出包）与 JIT（用到时现编）两条路径。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T13 <标题>`
