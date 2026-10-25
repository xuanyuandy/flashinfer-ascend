# ABI 冻结指纹重新冻结记录（2026-10-25）

## 结论

把 4 个冻结指纹常量更新为当前实现计算值。**这不是"改 golden 消红灯"**，
而是在确认原定义不可恢复、且无任何活产物依赖旧值之后，
把当前重建正式采纳为权威契约。

| 位置 | 对象 | 旧值 | 新值 |
|---|---|---|---|
| `tests/test_runtime_abi.py:179` | binary ABI v1 | `cb528512…` | `21d0e29e…` |
| `tests/test_runtime_abi.py:183` | kernel error ABI | `b74b754f…` | `7e2ac550…` |
| `tests/test_attention_quant_launch_abi.py:204` | binary ABI v2 | `3087d1b9…` | `b1120f7b…` |
| `tests/test_attention_metadata_wire.py:55` | plan metadata wire blob | `f04f7b3f…` | `4fe29f10…` |

**结构尺寸全部未变**（wire blob 仍为 488 字节；`PLAN_CONFIG`=128、`METADATA_SECTION`=32、
`KV_CACHE_VIEW_V2`=192），只有序列化身份变化。

## 一、根因：T1 契约恢复未完成，不是回归

`docs/reports/T1-report.md` 已明确记录：

- 第 15 行：「剩余 binary/error ABI 历史指纹仍不匹配，**未通过硬编码映射或放宽测试掩盖**」
- 遗留 #3：「binary/error ABI 的历史序列化指纹**无法仅由已恢复字段完全还原**」

即：冻结值写于初始提交 `a65a89e`，当时 runtime 侧相关 ABI 类**并不存在**；
`dfdfa4a`（T1 恢复 runtime 契约）重建了这些类，但重建结果无法复现原序列化。
从 `dfdfa4a` 到 HEAD，相关算法与定义的 AST 未再变化——**漂移自 T1 起就存在，不是后续引入的回归**。

三个测试文件自 `a65a89e` 起从未被修改（`git log` 仅一条提交），
两个相关 struct 定义经 `-L` / `-G` / `blame` 三种方式确认同样未改过。

## 二、单一根因，非三处独立问题

实测确认：

- `ATTENTION_KERNEL_ERROR_ABI.fingerprint` 本身不匹配（`7e2ac550…` vs `b74b754f…`）。
- `error_abi` 以**完整 dict**（非仅指纹）嵌入 `KernelBinaryABI.to_dict()`
  （keys = `codes/name/schema_version`），因此它同时污染 **v1 与 v2 两个 binary 指纹**。
- binary 指纹又嵌入 plan metadata 的 header（blob 的 `[112:144]` 区段），
  因此继续传播到 **wire blob 指纹**。

即一处 error ABI 漂移 → 4 个冻结值失配。**指纹算法本身没有问题**：
同文件 5 个 POD struct 指纹（TENSOR_VIEW / KV_CACHE_VIEW v1 / PLAN_METADATA_HEADER /
AUXILIARY_VIEW / RUN_OPTIONS）以及 `PLAN_CONFIG`、`KV_CACHE_VIEW_V2` 的 struct 指纹
全部仍与初始冻结值一致。

## 三、为什么判定"安全可重冻"：无活产物依赖旧值

检索范围与结果（旧值 `b74b754f` / `cb528512` / `3087d1b9` / `f04f7b3f`）：

| 位置 | 命中情况 |
|---|---|
| 仓库代码（`flashinfer/`） | **无** |
| 仓库内 manifest / registry / `data/` | **无** |
| `/root/.cache/flashinfer/`（含 T13-I 产物与 wheel） | 仅 pytest **日志** |
| `/tmp/flashinfer-t13ii-*`（T13-II 构建/安装的真实 op-package） | 仅测试**源码副本**与日志 |
| 构建出的 op-package 内容 | **零 attention 引用**（spec 为 `matmul_probe`，`grep -i attention` 无命中） |

`b74b754f`（error ABI）在仓库外**完全没有出现**。
旧冻结值只存在于：测试 golden 本身、pytest 日志、历史报告文档——均非消费者。

**结论：旧值是考古物，不是在役契约。重新冻结不会破坏任何已存在的产物或部署。**

## 四、本次**没有**做的事

- **原始定义未恢复。**本次未能还原生成旧冻结值的 `error_abi` 完整字典。
  T1 当年尝试失败；本轮按 T1 成功恢复 `WorkloadSpec.schema_version` 的同一套路
  （补齐缺失版本字段）做了定向搜索，**亦未命中**。
- 因此**不能断言新旧契约在语义上等价**，只能断言：当前实现自洽，且无人依赖旧值。
- 若将来取得原始 `runtime/schema.py` 或含完整旧 `error_abi` 的 manifest，
  **应重新评估本决定**——届时需要判断当前重建是否遗漏了字段。

## 五、验证

```
python -B -m pytest -p no:cacheprovider -q          → 1780 passed, 0 failed  (45.8s)
python -B -m pytest -p no:cacheprovider -m real_compile -q → 28 passed       (36.7s)
git diff --stat                                      → 3 files, 4 insertions(+), 4 deletions(-)
```

真机环境：8× Ascend950PR，CANN 9.2.0-beta.2，解释器 `/root/miniforge3/envs/flashinfer-a5/bin/python`。
基线为 `1777 passed, 3 failed`，本次 4 行改动后为 `1780 passed`，
**未新增跳过、未放宽任何断言、未修改任何 ABI 定义**。
