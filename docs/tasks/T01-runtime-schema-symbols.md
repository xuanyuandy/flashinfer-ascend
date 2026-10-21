# T1 · 补齐 10 个缺失的 runtime 基础定义

| | |
|---|---|
| **阶段** | P0 · 解除阻塞 |
| **执行序号** | 1 / 20 |
| **依赖** | — |
| **规模** | 大 |
| **状态** | `未开始` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] 完成并通过验收

---

## 阶段背景（P0 · 解除阻塞）

> 这一阶段不做，后面所有任务都无法验证。全仓 104 个 Python 模块里 72 个今天 import 就失败。

---

## 任务正文

**目标**：让全部 104 个模块可导入、98 个测试文件可收集。

**现状**：大量模块 import 了 `flashinfer.runtime` 里并不存在的符号。实际的
`flashinfer/runtime/schema.py` 只有 123 行、6 个定义。

**缺失符号清单**（全仓 grep 类/函数定义命中 0）：

| 符号 | 被引用次数 | 用途推断 |
|---|---|---|
| `SchemaError` | 1722 | 契约校验失败异常，应为 `ValueError` 或 `RuntimeError` 的子类 |
| `ArtifactRef` | — | 编译产物引用（路径 + 入口符号 + 格式） |
| `ArtifactFormat` | — | 产物格式枚举（`.so` / 算子包 / …） |
| `ArtifactKind` | — | 产物种类枚举 |
| `KernelBinaryABI` | — | 二进制 ABI 约定 |
| `KernelLaunchABI` | — | 下发 ABI 约定 |
| `KernelConstraints` | — | 形状/对齐等约束 |
| `KernelCapabilityBinding` | — | capability 与 kernel 的绑定关系 |
| `WorkspaceFormula` | — | workspace 大小的计算式 |
| `QuantSpec` | — | 量化参数描述 |

**改哪里**：`flashinfer/runtime/schema.py`、`flashinfer/runtime/__init__.py`

**做什么**：
1. 先用 `rg -n "SchemaError|ArtifactRef|..." flashinfer/ tests/` 把**所有使用点**收集出来，
   从使用方式反推每个符号的真实形状（字段名、构造签名、是否 frozen dataclass）。
   **不要凭空设计**——这些模块是对着某个版本的 schema 写的，要还原那个版本。
2. 按还原结果补进 `schema.py`，在 `runtime/__init__.py` 导出。
3. 跑全量测试，记录失败项——这是第一次能看到那 72 个模块的真实质量。

**验收**：
```bash
# 全部模块可导入
python3 -c "
import pathlib, importlib, sys
bad=[]
for p in sorted(pathlib.Path('flashinfer').rglob('*.py')):
    if '__pycache__' in str(p): continue
    m=str(p.with_suffix('')).replace('/','.').removesuffix('.__init__')
    try: importlib.import_module(m)
    except Exception as e: bad.append((m, type(e).__name__, str(e)[:80]))
print('failing:', len(bad)); [print(' ', *b) for b in bad]
"
# 期望 failing: 0

pytest --collect-only -q 2>&1 | tail -5
# 期望 0 errors
```

**产出**：一份「72 个模块复活后的测试失败清单」，供后续任务排期。

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T1 <标题>`
