# 本仓已支持接口清单（按能力域分类）

> 基线：`infer` @ `fbca115`（工作区含 P0-1/P0-2 改动）· 2026-10-25
> 实测环境：8× Ascend950PR · CANN 9.2.0-beta.2 ·
> 解释器 `/root/miniforge3/envs/flashinfer-a5/bin/python`

## 0. 口径：三个"支持"不是一回事

本文只回答**第 2 列**。混用口径是这类对标表最容易出的错。

| 口径 | 含义 | 本文如何处理 |
|---|---|---|
| ① CANN 侧有对标 kernel | 昇腾生态里存在功能等价的算子 | **不在本文范围**（那是厂商能力调研） |
| ② **本仓暴露了接口且有已绑定 runner** | `import flashinfer` 后可调用，dispatch 能选到真实后端 | **本文清单** |
| ③ 实测在 NPU 上跑通 | 有真机执行证据 | 本文标注「实测」列 |

**①≠②。** 例如 `flashinfer.sparse`、`flashinfer.comm`、`flashinfer.mla` 在 CANN 侧确有对应能力，
但**本仓根本不存在这三个模块**（`ModuleNotFoundError`）。把 ① 的"已覆盖"写进本仓能力表会严重高估。

统计口径：登记表共 **79 条 descriptor / 43 个 op**。
一个 op 通常有 2 条（CANN + REFERENCE 各一），7 个 attention op 只有 1 条。
**REFERENCE 是 host 参考实现，不是 NPU 加速路径**；下表「后端」列中的 CANN / Triton 才是真实加速后端。

## 1. 总览

| 能力域 | op 数 | 本仓状态 |
|---|---|---|
| 标准 attention | 8 | ✅ 有 CANN 后端 |
| GEMM | 5 | ✅ 有 CANN 后端 |
| 采样 | 10 | ✅ 有 CANN 后端 |
| RoPE | 6 | ✅ 有 CANN 后端 |
| 归一化 | 5 | ✅ 4 个 CANN + 1 个 Triton |
| 模型专用融合（激活） | 3 | ✅ 有 CANN 后端 |
| MoE / EP | 3 | ✅ 有 CANN 后端 |
| attention 结果合并（cascade） | 2 | ✅ 有 CANN 后端 |
| KV 管理（page） | 1 | ✅ 有 CANN 后端 |
| **合计** | **43** | |
| MLA | 0 | ❌ 无模块 |
| 稀疏 attention | 0 | ❌ 无模块 |
| 通信 | 0 | ❌ 无模块 |
| 量化 | 0 | 🟡 有 `QuantSpec` 契约，无算子 |
| GDN / KDA / Mamba | 0 | ❌ 无模块 |
| 训练辅助 | 0 | ❌ 无模块 |
| trace | 0 | 🟡 仅 attention 内部协议 trace，无公开模块 |
| autotuner | 0 | 🟡 有 `flashinfer/tuning/`，无 autotuner 公开接口 |

## 2. 已支持接口明细

「实测」列：`✔` = 本轮在真机上直接调通；`T` = 由通过的测试覆盖。

### 2.1 标准 attention（8 op）

| 公共接口 | op | 后端 | 实测 |
|---|---|---|---|
| `single_prefill_with_kv_cache` | `attention.single_prefill` | CANN | ✔ |
| `single_decode_with_kv_cache` | `attention.single_decode` | CANN | ✔ |
| `cudnn_batch_decode_with_kv_cache` | `decode.cudnn_batch_decode_with_kv_cache` | CANN / REFERENCE | T |
| `BatchDecodeWithPagedKVCacheWrapper` | `attention.batch_decode_paged` | CANN | T |
| `BatchPrefillWithPagedKVCacheWrapper` | `attention.batch_prefill_paged` | CANN | T |
| `BatchPrefillWithRaggedKVCacheWrapper` | `attention.batch_prefill_ragged` | CANN | T |
| （host 参考路径） | `attention.decode` | REFERENCE | T |
| （host 参考路径） | `attention.prefill` | REFERENCE | T |

另有 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`（图模式 decode wrapper）与
`*_with_jit_module` 变体（需调用方自带 JIT module，不走登记表）。

**入图边界**：79 条 descriptor 中仅 **1 条**声明可入图
（`cann/attention/single_decode_fia_v1`，`NEEDS_PARAM_UPDATE`），`CAPTURE_SAFE` 为 0。

### 2.2 GEMM（5 op）

| 公共接口 | op | 后端 | 实测 |
|---|---|---|---|
| `mm_bf16` | `gemm.mm_bf16` | CANN / REFERENCE | ✔ |
| `bmm_bf16` | `gemm.bmm_bf16` | CANN / REFERENCE | ✔ |
| `bmm_fp8` | `gemm.bmm_fp8` | CANN / REFERENCE | T |
| `gemm_fp8_nt_groupwise` | `gemm.gemm_fp8_nt_groupwise` | CANN / REFERENCE | T |
| `SegmentGEMMWrapper` | `gemm.segment_gemm` | CANN / REFERENCE | T |

### 2.3 采样（10 op）

| 公共接口 | 后端 | 实测 |
|---|---|---|
| `sampling_from_probs` | CANN / REFERENCE | ✔ |
| `sampling_from_logits` | CANN / REFERENCE | ✔ |
| `top_p_sampling_from_probs` | CANN / REFERENCE | ✔ |
| `top_k_sampling_from_probs` | CANN / REFERENCE | ✔ |
| `min_p_sampling_from_probs` | CANN / REFERENCE | ✔ |
| `top_k_top_p_sampling_from_probs` | CANN / REFERENCE | ✔ |
| `top_p_renorm_probs` | CANN / REFERENCE | ✔ |
| `top_k_renorm_probs` | CANN / REFERENCE | ✔ |
| `top_k_mask_logits` | CANN / REFERENCE | ✔ |
| `chain_speculative_sampling` | CANN / REFERENCE | T |

> 注：上游该接口在你那份表里标「未覆盖（无投机解码 chain sampling 融合算子）」，
> 但**本仓登记表里有 `sampling.chain_speculative_sampling` 且绑定了 CANN runner**。

### 2.4 RoPE（6 op）

| 公共接口 | 后端 | 实测 |
|---|---|---|
| `apply_rope` | CANN / REFERENCE | T |
| `apply_rope_inplace` | CANN / REFERENCE | T |
| `apply_rope_pos_ids` | CANN / REFERENCE | T |
| `apply_rope_pos_ids_inplace` | CANN / REFERENCE | T |
| `apply_rope_with_cos_sin_cache` | CANN / REFERENCE | T |
| `apply_rope_with_cos_sin_cache_inplace` | CANN / REFERENCE | T |

`tests/test_rope.py` + `tests/test_rope_pos_ids.py` 共 27 项通过。
**无** Llama 3.1 NTK / low-freq 缩放变体（`apply_llama31_rope*` 不存在）。

### 2.5 归一化（5 op）

| 公共接口 | 后端 | 实测 |
|---|---|---|
| `rmsnorm` | CANN / REFERENCE | ✔ |
| `fused_add_rmsnorm` | CANN / REFERENCE | ✔ |
| `gemma_rmsnorm` | CANN / REFERENCE | ✔ |
| `gemma_fused_add_rmsnorm` | **Triton** / REFERENCE | ✔ |
| `layernorm` | CANN / REFERENCE | ✔ |

> `gemma_fused_add_rmsnorm` 是**全仓唯一**走 Triton 后端的算子（CANN 侧确实缺 Add+Gemma 融合，
> 本仓用 triton-ascend 自己实现）。它有真机数值测试对齐 fp64。

### 2.6 模型专用融合 · 激活（3 op）

| 公共接口 | 后端 | 实测 |
|---|---|---|
| `silu_and_mul` | CANN / REFERENCE | ✔ |
| `gelu_and_mul` | CANN / REFERENCE | ✔ |
| `gelu_tanh_and_mul` | CANN / REFERENCE | ✔ |

### 2.7 MoE / EP（3 op）

| 公共接口 | op | 后端 |
|---|---|---|
| `fused_moe` | `fused_moe.cutlass_fused_moe` | CANN / REFERENCE |
| `fp8_block_scale_moe` | `fused_moe.trtllm_fp8_block_scale_moe` | CANN / REFERENCE |
| `fp8_per_tensor_scale_moe` | `fused_moe.trtllm_fp8_per_tensor_scale_moe` | CANN / REFERENCE |

另有 `RoutingMethodType`（枚举）与 `reorder_rows_for_gated_act_gemm`（无独立 op，属上述流程内部步骤）。
**EP / 专家并行分发没有接口**（依赖通信域，而通信域本仓为空）。

### 2.8 attention 结果合并 · cascade（2 op）

| 公共接口 | 后端 |
|---|---|
| `merge_state` | CANN / REFERENCE |
| `merge_states` | CANN / REFERENCE |

无 `merge_state_in_place`。

### 2.9 KV 管理 · page（1 op）

| 公共接口 | 后端 |
|---|---|
| `append_paged_kv_cache` | CANN / REFERENCE |

无 `append_paged_mla_kv_cache`、无 `get_batch_indices_positions`。

## 3. 本仓不存在的能力域

下列模块 `import` 直接 `ModuleNotFoundError`，**不是"部分覆盖"，是没有**：

`flashinfer.mla`、`flashinfer.sparse`、`flashinfer.comm`、`flashinfer.fp4_quantization`、
`flashinfer.gdn`、`flashinfer.trace`、`flashinfer.autotuner`、`flashinfer.logits_processor`、
`flashinfer.green_ctx`、`flashinfer.xqa`、`flashinfer.cute_dsl`、`flashinfer.profiler`

对应到分类：

| 分类 | 状态 | 说明 |
|---|---|---|
| MLA | ❌ | 无模块。MLA 相关的 paged append 也缺 |
| 稀疏 attention | ❌ | 无模块（CANN 侧有 `block_sparse_attention`，本仓未接） |
| 通信 | ❌ | 无模块。NVIDIA 专有部分（MNNVL/NVSHMEM/CUDA IPC）属**有意不对标**；但 HCCL 路径也未接 |
| 量化 | 🟡 | `flashinfer.quantization` 只有 `QuantSpec` 数据契约，**无 packbits 等算子**；无 fp4 量化模块 |
| GDN / KDA / Mamba | ❌ | 无模块 |
| 训练辅助 | ❌ | 无模块（本仓定位为推理） |
| trace | 🟡 | 无公开模块；`flashinfer/attention/protocol_trace.py` 是内部协议记录，非上游 trace 对标 |
| autotuner | 🟡 | 有 `flashinfer/tuning/` 子包与 manifest，但无 autotuner 公开接口；发布 tuning 库为空 |
| 实验后端 | 🟡 | 存在 `flashinfer/triton/`（1 个算子）、`flashinfer/cann/`（40 条 descriptor）；无 cute_dsl / cubin / TRT-LLM 后端 |

## 4. 与 CANN 对标调研表的关系

那份表的「NPU 支持情况 / CANN 对标实现」两列回答的是**口径 ①**：昇腾生态有没有等价 kernel。
本文回答**口径 ②**：本仓有没有接口和 runner。两者差距最大的几处：

| 接口 | CANN 调研表 | 本仓实际 |
|---|---|---|
| `flashinfer.sparse` | 已覆盖 | **无模块** |
| `flashinfer.comm.*`（约 40 项） | 部分覆盖 / 未覆盖 | **无模块** |
| `flashinfer.page.append_paged_mla_kv_cache` | 部分覆盖 | **无接口** |
| `flashinfer.fp4_quantization.*` | 多项已覆盖 | **无模块** |
| `flashinfer.rope.apply_llama31_rope*` | 部分覆盖 | **无接口** |
| `flashinfer.sampling.chain_speculative_sampling` | 未覆盖 | **有接口 + CANN runner** |

最后一行方向相反：调研表低估了本仓。

## 5. 证据与复现

```bash
# 登记表统计（43 op / 79 descriptor）
python -B -c "from flashinfer.runtime.registry import get_global_registry as g; \
  d=list(g().descriptors()); print(len(d), len({x.op for x in d}))"

# 真机接口冒烟（本轮 22/23 直接调通，apply_rope 由 27 项测试覆盖）
python -B -m pytest -p no:cacheprovider -q            # 1828 passed
python -B -m pytest -p no:cacheprovider -q -m real_compile   # 36 passed
```

**未验证**：各接口的全量 dtype / shape / layout 组合、性能、
非 Ascend950 设备、以及 batch wrapper 在 aclgraph 下的行为。
本文只声明「接口存在且可调通」，不声明「功能与上游逐参数等价」。
