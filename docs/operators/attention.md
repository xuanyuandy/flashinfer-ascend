# Attention 接口与实现范围

面向集成者：本页区分当前可调用的 single/batch CANN、独立 CATLASS FA 和 arch22 SFA，避免把同族接口当成同一实现或支持域。

## 公共接口

| 路线 | 入口 | 实现与边界 |
| --- | --- | --- |
| single prefill/decode | `single_prefill_with_kv_cache`、`single_decode_with_kv_cache` | CANN FIA；FP16/BF16，NHD/HND、GQA 等以公共校验和 runner 为准 |
| batch paged/ragged prefill | 对应 BatchPrefill wrapper 的 plan/run | 真实 Tensor 路径为 CANN；组合逐请求 gather/FIA/拼接，不等于完整 fused scheduler |
| batch paged decode | BatchDecode wrapper 的 plan/run | CANN 路径，计划与 workspace 由 wrapper 持有；名字带 CUDAGraph 不扩大图能力 |
| 独立 dense CATLASS | `single_prefill_with_kv_cache_catlass` | Ascend910B/Ascend910C/Ascend950、本仓 AOT，允许声明范围内的缺包 JIT |
| MLA sparse | `sparse_flash_attention` | Ascend910B/Ascend910C 共用 arch22、本仓 AOT-only；没有同 op 的 CANN/Triton 生产实现 |
| 分组 QKV 重排 | `permute_qkv(q, k, v, split_size)` | Ascend910B/Ascend910C，arch22 向量核 AOT |
| IndexerPrologue QW | `indexer_prologue_qw`（`prequantized=True` 复用同一入口） | Ascend910B/Ascend910C，单 mixed launch；dense 每次执行 weight DynamicQuant，prequantized 显式接收调用方 NZ weight/scale |
| IndexerPrologue K | `indexer_prologue_k` | Ascend910B/Ascend910C，单 mixed launch；原地更新调用方的独立-strided key/scale cache，返回 `None` |

普通 single 的 auto 会读取全局默认，普通 batch Tensor 的非 reference plan 分支固定请求 cann；它们不是任意后端的统一占位接口。完整规则见 [后端路由](../concepts/backend-routing.md)。

## CATLASS dense prefill

Q 为 `[S,H,128]`，K/V 为 `[T,H,128]`，隐含 batch=1；S/T 是 128..512 的 128 倍数，H 为 1..8。Q/K/V/out 为相同 FP16 或相同 BF16，连续物理 ND、逻辑 NHD。只支持非 causal、无 mask，不接收运行期长度 Tensor。

```python
output = flashinfer.single_prefill_with_kv_cache_catlass(q, k, v, backend="ascendc")
```

该入口与普通 `attention.single_prefill` 是不同逻辑 op，不会仅因数学相似就自动混选 CANN。实验 Triton FA 需要独立 registry 作用域；默认生产集合不因此变化。

## arch22 PermuteQkv

`flashinfer.permute_qkv(q, k, v, split_size)` 接收 `q[S,B,Nq,D]` 和 `k/v[S,B,Nkv,D]`。`split_size` 表示组数。每组顺序存放该组的 Q、K、V 头，返回一个连续的 `[Nq+2*Nkv,S,B,D]` 张量。例如 `Nq=4, Nkv=2, split_size=2` 时，输出头顺序为 `Q0,Q1,K0,V0,Q2,Q3,K1,V1`。

三个输入取相同的 FP16 或 BF16 类型，维度均为正数，存储为连续 ND。四维 NPU 张量可用 `torch_npu.npu_format_cast(t, 2)` 准备。`Nq` 须能被 `Nkv` 整除，`Nkv` 须能被 `split_size` 整除。arch22 还要求 `D` 为 16 的倍数，`Nq <= 4095`，输出最多 `2**31-1` 个元素。`S*B` 为偶数时至少两行 Q 需要放入 180 KiB 的 UB，为奇数时至少一行，每行占 `2*Nq*D` 字节。

`flashinfer.unpermute_qkv(x, n_k)` 接收连续的 `[split,S,B,heads_per_split,D]`。每个分组内依次存放 Q、K、V。`n_k` 是所有分组合计的 KV 头数，须能被 `split` 整除。返回的 Q、K、V 形状依次为 `[S,B,Nq,D]`、`[S,B,n_k,D]`、`[S,B,n_k,D]`，其中 `Nq = split * heads_per_split - 2*n_k`。FP16 和 BF16 均要求 D 按 16 对齐，一行完整的 Q/K/V 应能放入 180 KiB UB。

## arch22 SparseFlashAttention

公共入口为 `flashinfer.sparse_flash_attention` 和 `flashinfer.attention.sparse_flash_attention`。它返回 `(attention_out, softmax_max, softmax_sum)`，不是单个 Tensor。

- 计算为 MLA absorb：D=512、KV head=1，Q head 为 1..64 或 128；FP16/BF16。
- query 支持 BSND/TND，KV 支持 BSND/TND/PA_BSND；TND 需要对应前缀长度，分页 KV 需要页表和有效长度。
- 可选 query/key RoPE 成对提供，Dr=64；稀疏块大小为 1..128 的二次幂。
- sparse_mode 仅 0 或 3，分别为 full 与 bottom-right causal；attention_mode 固定 2，pre_tokens/next_tokens 均要求 INT64_MAX，当前不提供额外窗口裁剪。
- 输入与用户 out 必须连续、物理 ND；四维 TorchNPU 默认分配可能带 NCHW 标签，调用方显式准备格式，框架不改写输入或替换用户 out。
- 默认后两个 FP32 输出形状为 `(0,)`；启用 return_softmax_lse 后，BSND 辅助形状为 `[B,1,Sq,H]`，TND 为 `[1,Tq,H]`。

布局只接受 BSND/BSND、TND/TND、BSND/PA_BSND、TND/PA_BSND 四种 query/KV 配对。主要输入如下：

| 输入 | BSND | TND 或分页约定 |
| --- | --- | --- |
| query | `[B,Sq,H,512]` | TND 为 `[Tq,H,512]` |
| key/value | `[B,Sk,1,512]` | TND 为 `[Tkv,1,512]`；PA_BSND 为 `[blocks,block_size,1,512]` |
| sparse_indices | INT32 `[B,Sq,1,Ns]` | TND 为 `[Tq,1,Ns]`；索引是序列内 KV 稀疏块编号，尾部用 -1 填充 |
| actual_seq_lengths_query / kv | 可选 INT32 `[B]` 有效长度 | TND 为非递减前缀和，末项等于总 token；分页 KV 必须提供有效长度 |
| block_table | 不提供 | 分页使用 INT32 `[B,max_blocks]` 物理块号；block_size 为 16..1024 的 16 倍数，且能被 sparse_block_size 整除 |

`scale_value` 要求正有限数。调用前可用 `torch_npu.npu_format_cast(tensor, 2)` 显式准备 ND 存储，特别是四维输入、索引、RoPE 和用户 out；shape 连续本身不保证物理格式标签满足约定。

```python
attention_out, softmax_max, softmax_sum = flashinfer.sparse_flash_attention(
    query, key, value, sparse_indices, scale_value,
    query_rope=query_rope, key_rope=key_rope,
    sparse_mode=3, return_softmax_lse=True, backend="ascendc",
)
```

索引、长度与页表的值约束由公共 validate_values 校验；native ABI 的正尺寸约束保持不变，关闭辅助输出时由 runner 提供内部占位缓冲。当前图声明为 NOT_CAPTURABLE，不从 eager 或编译成功推导图支持。

A2/A3 使用相同公共接口与 runner，910B 包含各自的冻结原生产物；A3 真机数值未验证。

## arch22 IndexerPrologue

Dense QW 接口为：

```python
q, q_scale, w = flashinfer.indexer_prologue_qw(
    x, qr, wqb, ww, rope_sin, rope_cos, softmax_scale=128 ** -0.5,
)
```

`x[T,5120]`、`qr[T,1280]`、`wqb[4096,1280]`、`ww[32,5120]` 为连续 BF16 ND；RoPE 表为 FP32 `[T,64]`。返回 INT8 `q[T,32,128]`、FP16 `q_scale/w[T,32]`。activation/weight DynamicQuant、量化矩阵乘、RoPE、W projection 和 query quantization 均在同一次 mixed launch 内完成；dense 不缓存 weight preparation。

调用方已持有量化权重时使用同一入口并设置 `prequantized=True`：

```python
wqb_nz, wqb_scale = flashinfer.prepare_indexer_prologue_qw_weight(wqb)
q, q_scale, w = flashinfer.indexer_prologue_qw(
    x, qr, wqb_nz, ww, rope_sin, rope_cos,
    softmax_scale=128 ** -0.5,
    wqb_scale=wqb_scale,
    prequantized=True,
)
```

`wqb_nz` 是逻辑 `[1280,4096]` 的 INT8 FRACTAL_NZ，`wqb_scale` 是 FP32 `[4096]`。helper 是显式、无缓存的普通调用；forward 不做隐藏 weight 转换。QW 入口支持 `out=`，并返回调用方提供的原输出对象。

K 的 rich mutation 接口为：

```python
flashinfer.indexer_prologue_k(
    latent, wk, norm_weight, rope_cos, rope_sin, indices,
    k_cache, k_scale_cache, norm_eps=1e-6,
)
```

`latent/wk` 为 BF16 `[T,512]`/`[128,512]`，`norm_weight` 为 BF16 或 FP32 `[128]`，RoPE 表为 FP32 `[T,1,1,64]`。INT32 `indices[T,2]` 保存 `[block_id,block_offset]`，仅精确 `[-1,-1]` 表示 padding。`k_cache`/`k_scale_cache` 分别为 INT8 `[NB,NS,1,128]` 和 FP16 `[NB,NS,1,1]`，允许独立的非连续 block/slot stride，最后一维必须连续。该入口原地更新 cache、返回 `None`，不提供 `out=`；`T=0` 只做静态检查，不加载 AOT 或启动 kernel。重复有效 slot 不属于已声明的输入域。

两个入口的 descriptor 均为 `NOT_CAPTURABLE`，不把 eager 可执行或交叉编译成功解释为 graph capture/replay 支持。

## 支持与缺口

普通 CANN Attention 的登记范围主要是 Ascend950，个别兼容入口另有 Ascend910B/Ascend910C 声明；按 [逐项矩阵](../backends/overview.md) 核对，不按设备目录名外推。

混合 BatchAttention、量化 KV provider 和完整上游 MLA/sparse/POD wrapper 尚未形成默认 NPU 产品面。保留的 Host 合同不是可运行后端说明；接口与图扩展目标集中在 [规划](../roadmap.md)。

源码：[single 前端](../../flashinfer/_single_attention.py)、[batch 前端](../../flashinfer/attention/public_batch.py)、[CATLASS 声明](../../flashinfer/ascendc/attention/single_prefill_catlass/implementation.py)、[SFA 公共定义](../../flashinfer/operators/attention/sparse_flash_attention/operator.py)、[SFA 声明/runner](../../flashinfer/ascendc/attention/sparse_flash_attention/)。
