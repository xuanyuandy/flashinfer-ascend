# 扫描结果的读取方式

本目录由 `tools/audit_operator_surface.py` 与 `tools/build_operator_comparison.py` 生成。
范围、结论及官方文档见 [完整说明](../../operator-coverage.md)，调用方式和 UT 见 [生命周期说明](../../operator-call-patterns.md)。

| 文件 | 口径 |
| --- | --- |
| `upstream-api-comparison.csv` | 顶层及扫描清单内公共模块的 Python 函数/类导出，包含未在测试直接出现的补集 |
| `backend-and-experimental-comparison.csv` | 其他测试引用的可解析函数、类和方法；含私有、实验及下层实现 |
| `canonical-declarations.csv` | 公共视图按实际源码声明合并拼写；不推断数学等价 |
| `wrapper-lifecycles.csv` | 所有已识别 wrapper 的生命周期、签名和暴露层级；包含别名 |
| `test-file-coverage.csv` | 全部测试目录 Python 文件及摘要、引用；扫描不等于执行 |
| `dynamic-lookups.csv` | 非字面量属性查询；任意运行时名称不假装已解析 |
| `unresolved-references.csv` | 无法静态绑定的属性、装饰器附加方法或动态符号；不自动计作缺失算子 |
| `local-operators.md` / `.csv` | 当前本仓逻辑 op；CSV 含能力限制、backend、源码、Torch 入口等 |
| `local-public-interfaces.csv` | 本仓公开函数/类和到 op 的对应；未绑定 op 的框架类型不算 NPU 算子 |
| `local-descriptors.json` | 当前运行时登记快照，保留 capability 和图能力 |
| `summary.json` / `scan-manifest.json` | 计数、扫描模块、commit 和输入/源码 SHA-256 |

`candidate_group` 与 `official_source_ids` 指向 [人工审阅规则](../../operator-comparison-rules.json)。
它们是候选实现路线。`local-entry-limited` 只说明本仓存在受限公共路线，**不表示逐参数等价、所有产品已支持、图模式或性能已验证**。
`role_hint` 是函数名和源码类别辅助分类，歧义仍需按签名/源码审阅；helper、配置、资源类和生命周期方法不计作独立数学算子。

同一 op 的顶层/子模块导出、Torch functional/out，以及 wrapper 的 plan/run 都应保留各自接口契约；这些行不能相加作为物理 kernel 数量。
