# T1 · 补齐 10 个缺失的 runtime 基础定义

| | |
|---|---|
| **阶段** | P0 · 解除阻塞 |
| **执行序号** | 1 / 20 |
| **依赖** | — |
| **规模** | 大 |
| **状态** | `导入/收集门禁通过；ABI 指纹待修复` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] 完成并通过验收

---

## 阶段背景（P0 · 解除阻塞）

> 这一阶段不做，后面所有任务都无法验证。全仓 104 个 Python 模块里 72 个今天 import 就失败。

---

## ⚠ 范围边界（事后补记，已验证成立）

> **这段是在 T1 已经执行之后补进来的，保留它是因为它预言的事真的发生了。**
> 本应在动手前先交付「符号与字段对照表」、确认最小修改范围再改代码；
> 这一步被跳过，结果是 **T1 分三次提交**，后两次都是回头补漏掉的字段：
>
> | 提交 | 内容 |
> |---|---|
> | `dfdfa4a` | T1 恢复 runtime 契约（47 文件） |
> | `8d7037a` | 回头补量化分组校验与 manifest 绑定字段 —— `schema.py` **+5 行** |
> | `0581b19` | 回头补 workload 版本字段 —— `schema.py` **+3 行** |
>
> 对照表其实做了（`docs/reports/T1-contract-map.json`，15,045 行），
> 只是没有停下来等确认。**后续任务遇到同类情况，按下面的做法走。**

**列出的 10 个符号是当前能看见的，实际范围更大。**
import 在第一个缺失符号处就中断，后面的还没机会报出来。
现在按首个错误归类只有 5 类：

| 首个报错 | 阻塞模块数 |
|---|---|
| `SchemaError` | 66 |
| `ArtifactRef` | 2 |
| `ArtifactFormat` | 2 |
| `QuantSpec` | 1 |
| `triton` 未安装（**与 schema 无关，不要当成本任务范围**） | 1 |

补上 `SchemaError` 后，被它挡住的 `ArtifactKind` / `KernelBinaryABI` /
`KernelLaunchABI` / `KernelConstraints` / `KernelCapabilityBinding` /
`WorkspaceFormula` 才会浮出来。**所以这是一个迭代发现的过程，不是一次性补齐。**

**CLI 与入口也被阻塞**，不只是库内部：

```
python -m flashinfer --help   →  ImportError: cannot import name 'SchemaError'
import flashinfer.cli         →  同上
import flashinfer.parity      →  同上（从 .runtime.schema 导入）
```

两条导入路径都要满足：`flashinfer.runtime.schema`（定义）与
`flashinfer.runtime`（再导出）。

### 本任务的第一个交付物

**不是代码，是一份「符号与字段对照表」**：每个符号的全部使用点 →
反推出的字段名 / 构造签名 / 是否 frozen。
**交付后确认最小修改范围，再开始改代码。**

引用量供参考（全部 `定义 0`）：

| 符号 | 引用 | | 符号 | 引用 |
|---|---|---|---|---|
| `SchemaError` | 2154 | | `ArtifactFormat` | 25 |
| `QuantSpec` | 149 | | `ArtifactKind` | 24 |
| `ArtifactRef` | 36 | | `KernelBinaryABI` | 14 |
| `KernelLaunchABI` | 12 | | `WorkspaceFormula` | 10 |
| `KernelConstraints` | 8 | | `KernelCapabilityBinding` | 6 |

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

## 2026-10-21 自动执行记录

见 [T1 报告](../reports/T1-report.md) 与 [结构化失败清单](../reports/T1-failures.json)。全模块导入和测试收集已解除阻塞；保留 binary/error ABI 历史指纹失败，不将自动验收视为用户验收。
