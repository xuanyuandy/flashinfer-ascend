# 稀疏索引器（Lightning Indexer）

面向集成者：本页说明 `sparse_indexer.lightning_indexer` 的公共接口、选择语义与当前支持面。
实现由 gitcode `cann/ops-transformer` 仓 `experimental/attention/lightning_indexer`（!12993）迁移：
设备 kernel 与 host tiling 逻辑保持不变，接入本仓的算子声明、AOT 构建与测试体系。

## 接口

```python
flashinfer.lightning_indexer(
    query, key, weights, *,
    cur_seq_lengths_query=None, cur_seq_lengths_key=None, block_table=None,
    layout_query="TND", layout_key="PA_BSND",
    sparse_count=2048, kv_block_len=1, q_block_len=1,
    init_num=0, local_num=0, sparse_mode=3,
    pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
    return_value=False,
) -> (sparse_indices, sparse_values)
```

计算每个 query token 对全部 key 的加权 ReLU(QK) 头归约分数，按 `kv_block_len` 聚合成块分数后
做 TopK，返回选中块索引与分值。sink（`init_num`）与局部窗口（`local_num`）区域强制入选；
对角块经交换置于其逻辑位置。`sparse_values` 仅在 `return_value=True` 时写入实际分值，
否则保持全零缓冲。

| 参数 | 约束 |
| --- | --- |
| `query` | FP16/BF16，`TND`（当前唯一已验证的 query 布局），末维 D=128 |
| `key` | 与 query 同 dtype；`PA_BSND`（page 缓存，块大小为 16 的倍数且 ≤4096）或 `TND` |
| `weights` | FP32，`[Tq, N1]` |
| `cur_seq_lengths_query` / `cur_seq_lengths_key` | INT64 **累积前缀和** `[B+1]`；TND 必填 |
| `block_table` | INT32 `[B, max_blocks]`；`PA_BSND` 必填 |
| `sparse_count` | `[1, 4096]` 且为 128 的倍数 |
| `kv_block_len` | `1/2/4/8/16`；块分数为块内均值 |
| `init_num` / `local_num` | `[0,16]` / `[0,2048]`，须被 `kv_block_len` 整除 |
| `sparse_mode` | `0` 或 `3`（默认 3：下三角因果）；当前能力面只声明 3 |

`sparse_indices` 为 INT32，无效位置为 `-1`；`sparse_values` 与 query 同 dtype。
公共接口名与默认值与迁移源 torch schema 一致；`kv_block_len` 在原生 ABI 中对应属性名 `block_len`。

## 与官方 npu_lightning_indexer 的关系

torch_npu 内置 `torch.ops.npu.npu_lightning_indexer` 提供 DeepSeek DSA 官方版：
`actual_seq_lengths_*` 命名、`return_values`、无块聚合/sink/局部窗口参数、`layout_key` 默认
`BSND`。本算子是其实验扩展，二者不共用接口；调用方不应混用参数约定。

## 支持面与验证状态

| 项 | 值 |
| --- | --- |
| 实现 | `ascendc/sparse_indexer/lightning_indexer/native/Ascend910B`、`.../Ascend910C`（共用 arch22 源码） |
| dtype | float16 / bfloat16（query、key、values），weights FP32 |
| 布局 | query=TND；key=TND 或 PA_BSND |
| 执行 | AOT-only，`NOT_CAPTURABLE`；长度张量须 INT64，无隐式数值转换 |

**已验证**：Ascend910B 包在 910 系硬件上的真机数值测试——5 组用例（含 N1=64 生产头数、
`kv_block_len=4`、`return_value=True`）的 indices 逐 token 精确匹配 CPU golden，values
按位一致；负向校验（`sparse_count` 越界、`q_block_len>1`、布局不匹配）按预期拒绝。
Ascend910C 包完成交叉编译，真机数值未验证。

**已知边界**：BSND 布局未声明——迁移源的 tiling 对 BSND 长度按 `[B]` 逐序列校验，而设备
kernel 无条件按 `[B+1]` 累积差分读取，二者矛盾，不存在可正确执行的 BSND 调用。
`sparse_mode=0`、`pre_tokens`/`next_tokens` 非默认值未经验证，不在声明能力内。

数值参考见 `flashinfer/operators/sparse_indexer/lightning_indexer/cases.py`（CPU golden
复刻迁移源测试的选择语义，含哨兵分值模型）；真机用例见
[tests/operators/test_lightning_indexer.py](../../tests/operators/test_lightning_indexer.py)。
