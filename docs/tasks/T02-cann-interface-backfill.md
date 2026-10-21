# T2 · 补齐 32 条「CANN 有、本仓没接」的接口

| | |
|---|---|
| **阶段** | P1 · 接口对齐（第一步任务） |
| **执行序号** | 3 / 20 |
| **依赖** | —（验证需 T1） |
| **规模** | 大 |
| **状态** | `未开始` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] **T2.1** sampling（10 个接口）— 性价比最高
- [ ] **T2.2** Attention（4 个接口 + 2 个占位转正）
- [ ] **T2.3** rope（5 个接口）
- [ ] **T2.4** GEMM（4 个接口）
- [ ] **T2.5** activation（3 个接口）
- [ ] **T2.6** 其余（cascade / page / sparse / fp4，各 1~2 条）

---

## 阶段背景（P1 · 接口对齐（第一步任务））

> 这是本仓的主线职责。依据是 147 条算子覆盖表与本仓登记表的三列联表。
> 当前：本仓实际接入 15 条（全表 10%）；CANN 侧「已覆盖」46 条。

---

## 任务正文

**依赖**：无（可与 T1 并行，但验证需要 T1）

**总体**：这 32 条约需 15 个 CANN 算子，其中 **3 个本仓已在调用**。
按性价比分成 6 个子任务，**建议按序执行**。

每个子任务的统一做法：
1. 在对应模块（`flashinfer/sampling.py` 等）按**上游同名、同参数名、同参数顺序**定义接口。
   参数顺序以 `flashinfer-ori` 对应函数为准，逐字核对。
2. 实现里调 `torch_npu` 对应算子。
3. 在 `flashinfer/cann/` 下注册 `KernelDescriptor`，`backend=Backend.CANN`，
   `capability` 如实声明 dtype / layout / soc_models。
4. 写对拍测试：与上游同输入下的数值一致性（或在无 GPU 环境下与 reference 实现对拍）。
5. **更新对标账本**对应条目状态。

#### T2.1 · sampling（10 个接口）— 性价比最高

**需要的 CANN 算子**：`top_k_top_p_sample` / `top_k_top_p_sample_v2`（1~2 个算子覆盖 10 个接口）

**上游接口**（在 `flashinfer-ori` 中确认准确签名后再写）：
`sampling_from_probs` / `sampling_from_logits` / `top_p_sampling_from_probs` /
`top_k_sampling_from_probs` / `top_k_top_p_sampling_from_probs` /
`min_p_sampling_from_probs` / `top_p_renorm_probs` / `top_k_renorm_probs` /
`top_k_mask_logits` / `chain_speculative_sampling`

**先核实**：`top_k_top_p_sample_v2` 是否支持 MinP。不支持则 `min_p_sampling_from_probs`
单独处理或明确标注未实现。

**新建文件**：`flashinfer/sampling.py`（本仓目前该域完全为空）

#### T2.2 · Attention（4 个接口 + 2 个占位转正）

**需要的 CANN 算子**：`npu_fused_infer_attention_score`（**本仓 paged decode 已在调**）、
`npu_prompt_flash_attention`

**重点**：
- `BatchPrefillWithPagedKVCacheWrapper` 和 Ragged 版——**上游两个主力 prefill 入口，
  vLLM/SGLang 在用，本仓完全没有**。这两个是 P0 优先级。
- `single_prefill_with_kv_cache` / `single_decode_with_kv_cache`——
  本仓目前**只注册了 reference 实现**，要换成真实 CANN 实现。

**注意**：`npu_fused_infer_attention_score` 按 `Q_S == 1` 自动切 IFA / PFA 分支，
所以 decode 与 prefill 可以共用同一个算子调用点，只是参数不同。

#### T2.3 · rope（5 个接口）

**需要的 CANN 算子**：`npu_apply_rotary_pos_emb` / `rotary_position_embedding` /
`rope_with_sin_cos_cache`（3 个；本仓已接同族的 `npu_rotary_mul`）

#### T2.4 · GEMM（4 个接口）

**需要的 CANN 算子**：`npu_quant_batch_matmul_v4` + `npu_grouped_matmul`（后者本仓已在调）

#### T2.5 · activation（3 个接口）

**需要的 CANN 算子**：`npu_swiglu`（已在调）+ `gelu_mul` 类算子

**上游接口**：`silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul`
（`silu_and_mul` 是常用算子，不是边缘接口）

#### T2.6 · 其余（cascade / page / sparse / fp4，各 1~2 条）

- `merge_states` — 走 `npu_attention_update`（**本仓 cascade 已在调**），
  只差支持 TensorList 入参
- `append_paged_kv_cache` — 走 `npu_scatter_pa_kv_cache`
  （**先核实确切签名**）
- 其余按覆盖表逐条处理

**T2 整体验收**：
```bash
# 登记表条目数显著增加，且新增条目 backend 为 cann
python3 -c "
from flashinfer.runtime.registry import get_global_registry
r = get_global_registry()
ds = r.descriptors()
print('总条目:', len(ds), '算子数:', len({d.op for d in ds}))
from collections import Counter
print(Counter(d.backend.value for d in ds))
"
pytest tests/ -q
```

---

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T2 <标题>`
