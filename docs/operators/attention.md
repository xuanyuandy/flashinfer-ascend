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
| C4A4 分页 GQA | `quant_flash_attn_metadata`、`quant_flash_attn` | 面向 Ascend950 的 AICPU 调度与 Ascend C AOT 实现 |
| 分组 QKV 重排 | `permute_qkv(q, k, v, split_size)` | Ascend910B/Ascend910C，arch22 向量核 AOT |

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

## arch35 C4A4 QuantFlashAttention

`flashinfer.quant_flash_attn_metadata` 生成设备侧 INT32 FA/FD 调度表。主算子 `flashinfer.quant_flash_attn` 接收兼容的调度表，返回 BF16 Attention 和可选的 FP32 LSE。两个算子各自构建并发布 Ascend950 AOT 包。

下面的 shape 使用 PyTorch 张量维度。`B` 是请求数，`Tq` 是整个 batch 拼接后的 Q token 数，`Hq` 和 `Hkv` 是 Q 和 KV 头数，`G = Hq / Hkv` 是每个 KV 头对应的 Q 头数，`D` 是逻辑 head dim。`P` 是 KV cache 的物理页总数，`S` 是每页 token 数，`Pmax` 是每个请求在页表中最多可引用的页数。

`D` 支持 128 和 256。`S` 支持 64、128、256、512 和 1024。`B` 最大为 65535，`Hq` 和 `Hkv` 各最多为 256，`Hq` 须能被 `Hkv` 整除，`G` 最大为 64。FP4×2 对应 `torch.float4_e2m1fn_x2`，每字节存两个 FP4 值。E8M0 对应 `torch.float8_e8m0fnu`。

### 主算子 `quant_flash_attn`

`q`、`k`、`v` 和三个 descale 可以按位置传入。其余参数按名字传入。

| 输入 | dtype | shape | 是否必填及用途 |
| --- | --- | --- | --- |
| `q` | FP4×2 | `[Tq,Hq,D/2]` | 必填。按 TND 排布。 |
| `k`、`v` | FP4×2 | 见下方 KV 布局表 | 必填。分页 K、V 缓存使用相同 shape。 |
| `q_descale` | E8M0 | `TND` 为 `[Tq,Hq,D/64,2]`，`N2TGD` 为 `[Hkv,Tq,G,D/64,2]` | 必填。形状由 `layout_q_descale` 决定。 |
| `k_descale`、`v_descale` | E8M0 | 见下方 KV 布局表 | 必填。分别使用 K、V 对应的 scale 排布。 |
| `block_table` | INT32 | `[B,Pmax]` | 必填。每行列出一个请求使用的物理页。有效页号须指向当前 KV cache，页表须覆盖该请求的全部有效 KV token。`S * Pmax` 须在正 INT32 范围内。 |
| `cu_seqlens_q` | INT32 | `[B+1]` | 必填。每个请求的 Q 累计长度，从 `0` 开始，末项为 `Tq`。相邻两项的差值是该请求对应的 Q token 数。 |
| `seqused_kv` | INT32 | `[B]` | 必填。每个请求实际参与计算的 KV token 数。 |
| `metadata` | INT32 | `[1,C]` | 必填。与本次输入匹配的 FA/FD 调度表，通常使用下方生成器的返回值。 |
| `seqused_q` | INT32 | `[B]` | 可选，默认 `None`。提供时指定每个请求实际参与计算的 Q token 数，否则使用 `cu_seqlens_q` 的差值。 |
| `attn_mask` | INT8 | `[2048,2048]` | `mask_mode=3` 时传入 causal 模板，`mask_mode=0` 时省略。计算时按 Q 和 KV 的有效长度确定右下 causal 边界。 |

| `layout_kv` | `k`、`v` 的 shape | `k_descale` 的 shape | `v_descale` 的 shape |
| --- | --- | --- | --- |
| `PA_BBND` | `[P,S,Hkv,D/2]` | `[P,S,Hkv,D/64,2]` | `[P,S/64,Hkv,D,2]` |
| `PA_BNBD` | `[P,Hkv,S,D/2]` | `[P,Hkv,S,D/64,2]` | `[P,Hkv,S/64,D,2]` |
| `PA_NZ` | `[P,Hkv,D/64,S,32]` | `[P,Hkv,S/16,D/64,16,2]` | `[P,Hkv,D/16,S/64,16,2]` |

所有 Tensor 位于同一 NPU，物理存储格式为 ND，包括按 `PA_NZ` 逻辑排布的缓存。Q、Q descale、索引、长度、mask 和 metadata 要求连续存储。KV 缓存及其 descale 的内层维度要求连续。`PA_BBND` 可在物理页轴上保留 stride，`PA_BNBD` 和 `PA_NZ` 可在物理页轴和 KV 头轴上保留 stride。

| 选项 | 类型 | 默认值 | 取值及用途 |
| --- | --- | --- | --- |
| `quant_mode` | `int` | `5` | 固定为 `5`，选择 C4A4。 |
| `quant_schema` | `int` | `2` | `0` 为 OCP，`1` 为 CX，`2` 为 CX Delta。 |
| `softmax_scale` | `float` | `1.0` | 作用于 QK 分数。须为可用 FP32 表示的有限正数，常用 `D**-0.5`。 |
| `mask_mode` | `int` | `0` | `0` 为无 mask，`3` 为右下 causal。 |
| `win_left`、`win_right` | `int` | 均为 `-1` | 两项均固定为 `-1`，表示无额外窗口限制。 |
| `max_seqlen_q`、`max_seqlen_kv` | `int` | 均为 `-1` | 取 `-1` 或正 INT32。`-1` 表示未指定，每个请求的有效长度由长度 Tensor 提供。 |
| `layout_q`、`layout_out` | `str` | 均为 `TND` | Q 和 attention 输出固定使用 TND。 |
| `layout_q_descale` | `str` | `TND` | `TND` 对应 prefill 路径，`N2TGD` 对应 decode 路径。 |
| `layout_kv` | `str` | `PA_BBND` | 支持 `PA_BBND`、`PA_BNBD` 和 `PA_NZ`。 |
| `return_softmax_lse` | `bool` | `False` | 为 `True` 时计算并返回每个 Q 头和 token 对应的 LSE。 |
| `out` | 两个 Tensor 组成的 tuple 或 list | `None` | 按 `(attn_out, softmax_lse)` 提供两个输出缓冲。关闭 LSE 时第二项传 FP32 空 Tensor。 |

返回值始终是 `(attn_out, softmax_lse)`。

| 输出 | dtype | shape | 含义 |
| --- | --- | --- | --- |
| `attn_out` | BF16 | `[Tq,Hq,D]` | TND attention 结果。 |
| `softmax_lse` | FP32 | 启用时为 `[Hq,Tq]`，关闭时为 `[0]` | `softmax_lse[h,t]` 是头 `h` 在 Q token `t` 上的 log-sum-exp。 |

### 调度生成器 `quant_flash_attn_metadata`

五个必填参数可以按表中顺序传入。

| 输入或选项 | 类型或 shape | 是否必填及用途 |
| --- | --- | --- |
| `num_heads_q`、`num_heads_kv` | `int` | 必填。分别填写 `Hq` 和 `Hkv`。 |
| `head_dim` | `int` | 必填。填写逻辑维度 `D`，支持 `128` 和 `256`。 |
| `cu_seqlens_q` | INT32 `[B+1]` | 必填。与主算子使用相同的 Q 累计长度，`B` 从该 Tensor 推导。 |
| `seqused_kv` | INT32 `[B]` | 必填。与主算子使用相同的 KV 有效长度。 |
| `seqused_q` | INT32 `[B]` | 默认 `None`。主算子指定了有效 Q 长度时，同步传给生成器。 |
| `max_seqlen_q`、`max_seqlen_kv` | `int` | 均默认 `-1`，表示未指定。指定时须为正 INT32。 |
| `head_dim_v` | `int` 或 `None` | 默认 `None`，即使用 `head_dim`。指定时须等于 `head_dim`。 |
| `mask_mode` | `int` | 默认 `0`，支持 `0` 和 `3`。与主算子一致。 |
| `layout_q`、`layout_out` | `str` | 均默认并固定为 `TND`。 |
| `layout_q_descale` | `str` | 默认 `TND`，也支持 `N2TGD`。与主算子一致。 |
| `layout_kv` | `str` | 默认 `PA_BBND`，也支持 `PA_BNBD` 和 `PA_NZ`。与主算子一致。 |
| `quant_mode` | `int` | 默认并固定为 `5`。 |
| `out` | INT32 `[1,C]` | 默认 `None`。可传单个输出 Tensor，或只包含该 Tensor 的 tuple 或 list。 |

| 输出 | dtype | shape | 含义 |
| --- | --- | --- | --- |
| `metadata` | INT32 | `[1,C]` | 设备侧 FA/FD 调度表，直接传给主算子的 `metadata`。 |

将 `16 * (1 + 108 * B * Hq)` 向上对齐到 4096 个元素得到 `C`。108 是用于分配容量的 AIC 和 AIV 合计核数上限。AICPU 按实际有效核数写入表头和调度记录。批次重排或有效长度变动后，重新生成调度表。

### 两个接口的调用选项

| 选项 | 默认值 | 用法 |
| --- | --- | --- |
| `backend` | `None` | 由公共路由选择实现，可显式指定 `"ascendc"`。 |
| `kernel_id` | `None` | 可指定完整 ID。主算子为 `ascendc/attention/quant_flash_attn/native/Ascend950`，生成器为 `ascendc/attention/quant_flash_attn_metadata/native/Ascend950`。 |

省略 `out` 时由接口分配输出。传入的每个输出缓冲须匹配表中的 dtype 和 shape，并与输入位于同一 NPU。输出缓冲使用连续 ND，且各输出与其他参数的存储区域不得重叠。

```python
metadata = flashinfer.quant_flash_attn_metadata(
    num_heads_q, num_heads_kv, head_dim, cu_q, used_kv, layout_kv="PA_BNBD",
)
output, lse = flashinfer.quant_flash_attn(
    q, k, v, q_descale, k_descale, v_descale,
    block_table=page_table, cu_seqlens_q=cu_q, seqused_kv=used_kv,
    metadata=metadata, layout_kv="PA_BNBD", softmax_scale=head_dim**-0.5,
    return_softmax_lse=True, backend="ascendc",
)
```

先用一次普通调用完成 Torch 算子注册，再用 `torch.compile(fullgraph=True)` 将 metadata 和主 QFA 放在同一张 PyTorch 图中。Ascend950 上已验证 Inductor。ACLGraph 可捕获使用固定输入和 `out` 的 `quant_schema=2` 主算子，metadata 在 capture 外生成。 [ACLGraph 调用条件](../npu_graph_extensions.md)

## 支持与缺口

普通 CANN Attention 的登记范围主要是 Ascend950，个别兼容入口另有 Ascend910B/Ascend910C 声明；按 [逐项矩阵](../backends/overview.md) 核对，不按设备目录名外推。

混合 BatchAttention、量化 KV provider 和完整上游 MLA/sparse/POD wrapper 尚未形成默认 NPU 产品面。保留的 Host 合同不是可运行后端说明；接口与图扩展目标集中在 [规划](../roadmap.md)。

源码：[single 前端](../../flashinfer/_single_attention.py)、[batch 前端](../../flashinfer/attention/public_batch.py)、[CATLASS 声明](../../flashinfer/ascendc/attention/single_prefill_catlass/implementation.py)、[SFA 公共定义](../../flashinfer/operators/attention/sparse_flash_attention/operator.py)、[SFA 声明/runner](../../flashinfer/ascendc/attention/sparse_flash_attention/)。
