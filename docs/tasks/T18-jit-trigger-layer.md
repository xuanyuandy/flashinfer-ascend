# T18 · 统一的多后端 JIT 触发层，明确 kernel ↔ 产物/符号映射

| | |
|---|---|
| **阶段** | P2+ · 选型与编译的分层 |
| **执行序号** | 8 / 20 |
| **依赖** | T1 T6.1 |
| **规模** | 大 |
| **状态** | `未开始` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] 完成并通过验收

---

## 阶段背景（P2+ · 选型与编译的分层）

> 这三条补的是「JIT 编译一整套」和「选取性能最优实现」这两个诉求里
> **前面阶段没覆盖到**的部分。T6 加的是**字段**，这里加的是**分层结构**。

---

## 任务正文

**依赖**：T1、T6.1

**要解决的问题**：编译产物**不只有 Ascend C 一种**。Triton（triton-ascend）、
Ascend C、外部仓的预编译产物、未来可能的其它语言——**触发编译的方式各不相同，
但底层都是 kernel**。需要一层统一抽象，否则每接一种后端就要在 dispatch 里加一条分支，
回到上游「636 处 `if backend ==`」的老路。

**现状**：
- `flashinfer/jit/` 共 7 个文件 1,146 行，**只有 `cache_key.py`（63 行）能 import**，
  其余全部卡在 `SchemaError` / `ArtifactFormat` 缺失上
- `JitModuleLoader`（`jit/loading.py`）是个**纯 `Protocol`，一行实现都没有**
- 全仓 `subprocess` 命中 0 处（**无法拉起编译器**）、`ctypes` 命中 0 处（**无法装载 .so**）
- `jit/core.py` 全是元数据构造，没有一行触发编译

**上游对应物**（可作为形状参考，`flashinfer-ori/flashinfer/jit/`，**111 个文件、40,888 行**）：
- `JitSpec` —— 一份「要编什么」的声明，每个后端有自己的 `gen_jit_spec`，
  但**全部归约到同一个 `build_and_load()` 入口**
- `JitSpecRegistry` —— 全局登记所有 JitSpec
- `JitSpecStatus` —— 统一描述产物状态：
  `name` / `is_compiled` / `library_path` / `sources` / `needs_device_linking`

**做什么**：
1. 定义统一的「编译单元」抽象（对标 `JitSpec`）：声明源码、目标 SOC、编译选项、
   产物期望路径与入口符号。
2. **每个后端只提供「如何生成这份声明」和「如何触发自己的编译器」两个钩子**，
   其余（缓存、锁、原子替换、加载、登记）由统一层承担。
3. 明确 **kernel_id ↔ 产物文件 ↔ 入口符号**的三元映射，并让它可查询
   （类似 `JitSpecStatus`，能回答「这个 kernel 现在编了没有、产物在哪、符号叫什么」）。
4. 实现真正的编译触发（`subprocess`）与产物装载（`ctypes` 或 torch 扩展加载）。

**注意**：设计文档里关于编译缓存、文件锁、原子替换的描述有 100 多行，
**对应实现是 0 行**。这些要在本任务一并落地，不要只做「能编出来」。

**验收**：
```bash
# 能查询任意 kernel 的产物状态
python3 -c "
from flashinfer.jit import describe_artifact   # 名字按实际实现
print(describe_artifact('<某个 jit kernel_id>'))
"
# 至少两种后端（Triton + Ascend C）走同一个入口完成「声明→编译→加载→登记」
pytest tests/ -q -k "jit"
```

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T18 <标题>`
