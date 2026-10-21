# T2 · 补齐 32 条「CANN 有、本仓没接」的接口

| | |
|---|---|
| **阶段** | P1 · 接口对齐（第一步任务） |
| **执行序号** | 3 / 20 |
| **依赖** | —（验证需 T1） |
| **规模** | 大 |
| **状态** | `执行中：T2.1 已通过自动验收` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [x] **T2.1** sampling（10 个接口）— 性价比最高
- [ ] **T2.2** Attention（4 个接口 + 2 个占位转正）
- [ ] **T2.3** rope（5 个接口）
- [ ] **T2.4** GEMM（4 个接口）
- [x] **T2.5** activation（3 个接口）
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

## T2.1 执行记录（2026-10-21）

- 10 个接口按固定上游 commit 的名称、参数顺序、默认值与注解声明，注册 10 个 CANN 与 10 个显式 reference descriptor，公开顶层导出。
- A5 原生算子与 MinP 已实测可用。原生 TopK/TopP 会按索引切并列项；本实现先用 NPU 张量算子按上游阈值过滤，再调原生采样，避免该静默差异。
- 14 项测试通过，含 FP16/BF16/FP32、并列阈值、indices、无效行、随机分布与本地复现、speculative 前缀和独立接受计数。
- CUDA Philox 的 seed/offset 流尚未实现，显式报错；随机序列跨硬件不同，已逐条登记 reason。该限制使随机接口标为 intentionally_different，不能宣称完整 RNG drop-in。
- 日志：`work_items/reports/execution/steps/T2.1-tests-final.log`；账本增量：`T2.1-parity-entries.json`。

## T2.2 首批执行记录（2026-10-21）

- single prefill、paged/ragged prefill 已接真实 CANN FIA；single decode 的 CANN runner 已用私有规范化入口对拍，公开形状和参数纠偏归 T4.2 单独提交。
- 消除前序 T1 为收集测试而加的 prefill 与 JIT 声明：现已真实执行；JIT 入口只承诺注入模块的调用 ABI，不冒充 T18 的编译器实现。
- 按固定上游对象恢复完整签名，发现旧测试缺 variant_owns_mask/uniform_q_len/lse_base/lse_layout 等参数，已根据源码更新断言；LSE 对外为 log2，wrapper 可选 ln/NH/HN。
- 真实 A5 对拍 21 项通过：FP16/BF16、64/128 维、GQA、NHD/HND、因果、分页末页与非顺序页、ragged、调用方 out/LSE。
- A5/CANN 9.2 的 BNSD masked GQA 有部分 LSE 未正确写出；探测证实 BSND 路径误差约 2.4e-7，选择 BSND 后保留原精度断言通过。日志 `T2.2-lse-probe.log`。
- eager 每请求收集 KV 与调用 FIA；未声明图捕获或性能最优。NPU graph/最大 workspace 查询留 T9/T10，显式拒绝；不把未知 workspace 声称为零。
- 单独归属：T2.2 增加 prefill 与 backend/JIT 路径；T4.2 修公开 single decode；T4.4 修 decode wrapper；T4.6 统一 PDL。原有有意差异与新增限制均登记。

- prefill/JIT/provider/量化接入定向回归：48 项通过（`T2.2-prefill-final.log`）；固定源码签名由 `check_prefill_signatures.py` 单独核验。

## T2.5 执行记录

三个 activation 接口及顶层导出已接入 npu_swiglu/npu_gelu_mul，注册 3 个 CANN 与 3 个显式 reference descriptor；FP16/BF16/FP32、6/128/4096 尾维、out、PDL 对拍 28 项通过。gelu 的 erf/tanh 两种模式单独调用；大尺寸按原生算子限制分块。

T2.3/T2.4 缺少原覆盖表中的确切 5/4 个接口名；已继续查找并请求原表路径，保留待核实状态，先推进清单中有明确名字的单元。

## T2.6 明确接口执行记录

- `merge_states` 与 `append_paged_kv_cache` 已新增并注册 CANN/reference，25 项真实 A5 对拍通过，涵盖 1/3/17/100 个状态、全无效与部分无效状态组、FP16/BF16/FP32、两种布局与 packed/分离 cache。
- 本机 attention_update 每次 TensorList 必须小于 16；用 float32 原生合并树适配，保留精度断言。
- scatter API 的准确关键字是 `cache_mode="Norm"`（ND）；默认 PA_NZ 不能直接写上游 ND cache。非连续视图通过 ND staging 写回，保持调用方 cache 地址与别名。
- 检出相邻历史差异：旧 `merge_state` 使用自然对数；按全局约束登记为有意不同，文档注明，未在该任务顺手改行为。
- sparse/fp4 的确切剩余接口仍依赖未提供的原覆盖表，本子任务不勾选整项完成。
