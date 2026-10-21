> 历史快照：本文件保留最初收到的任务原文。
> 当前任务、执行顺序和进度以 [`../docs/TASKS.md`](../docs/TASKS.md)
> 及 [`../docs/tasks/`](../docs/tasks/) 为准。本文件不再维护任务状态。

# flashinfer-npu 对标与演进任务清单

> 这份清单是给执行 agent 用的工作单。每条任务都写了**依赖、改哪里、验收标准**。
> 架构背景与论证见配套的架构说明文档；这里只写「做什么」。
>
> 基线：本仓 `a65a89e` · 上游 `flashinfer-ori @ 289557772` · vllm-ascend 本地检出
> 最后更新：2026-10-21

---

## 怎么用这份清单

1. **按阶段顺序执行。** P0 不完成，P1 之后的任务大部分无法验证（七成模块打不开）。
2. **同一阶段内的任务可并行**，除非任务里写了 `依赖：`。
3. **每条任务独立可验收。** 验收标准是可执行的命令或可检查的事实，不是「看起来对了」。
4. **不要扩大范围。** 如果执行中发现相邻问题，记录下来，不要顺手改——这份清单的顺序是按依赖排的。

### 全局约束

- **对标优先于优化。** 任何与上游行为不一致的地方，要么改成一致，要么在对标账本里显式登记为「有意不同」并说明原因。不允许存在未登记的静默差异。
- **不可实现的东西要明确报错，不要静默降级成不同的结果。** 例外：上游自己就是静默降级的行为（见 T4.6）。
- **新增接口必须同时更新对标账本**（`flashinfer/data/*.json`）。
- 提交信息用中文，一条任务一个提交，标题带任务号，例如 `T2.1 接入 sampling 的 10 个接口`。

---

# P0 · 解除阻塞

> 这一阶段不做，后面所有任务都无法验证。全仓 104 个 Python 模块里 72 个今天 import 就失败。

### T1 · 补齐 10 个缺失的 runtime 基础定义

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

# P1 · 接口对齐（第一步任务）

> 这是本仓的主线职责。依据是 147 条算子覆盖表与本仓登记表的三列联表。
> 当前：本仓实际接入 15 条（全表 10%）；CANN 侧「已覆盖」46 条。

### T2 · 补齐 32 条「CANN 有、本仓没接」的接口

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

### T3 · 把 46 条 NVIDIA 专有接口标成「有意不同」

**依赖**：无。**工作量极小，但能让 T2 的范围收敛**，建议最先做。

**范围**：
- comm 域 46 条中的 NVIDIA 互联绑定：MNNVL、NVSHMEM、CUDA IPC、Lamport 协议
  （昇腾对应物是 HCCL，是另一套东西，不是「缺口」）
- `green_ctx`（CUDA SM 切分）
- `shuffle_matrix_*`（CUTLASS 布局）
- NVFP4 的 swizzle 排布相关接口

**做什么**：在对标账本（`flashinfer/data/api_parity.json` 及
`attention_api_parity.json`）中把这些条目的
`semantic_status` 设为 `intentionally_different`，并写明原因。
**不要**把它们留在「待补」列表里。

**验收**：账本中 `intentionally_different` 条目数 ≥ 46，且每条带 `reason` 字段。

---

### T4 · 修掉会静默出错的对标缺陷

**依赖**：无

> 这些不挡开发，但会让下游拿到**错误结果**。对标是本仓的主要职责，这类缺陷最伤。
> 按危险程度排序。

#### T4.1 · MoE 路由枚举取值与上游不一致 ⚠ 最高危

**文件**：`flashinfer/fused_moe.py:54-70`

**问题**：本仓 `Unspecified = 5`，而**上游 `flashinfer/tllm_enums.py:10-38` 的 `5` 是 `TopK`**，
上游的 `Unspecified` 是 `10`。配置在两仓之间传递时会**静默选成错误的专家路由算法**。
更糟的是本仓该枚举的 docstring 写着「取值一致」——这句话是错的。

**上游完整定义**（`flashinfer-ori/flashinfer/tllm_enums.py`）：
```
Default=0, Renormalize=1, DeepSeekV3=2, Llama4=3, RenormalizeNaive=4,
TopK=5, SigmoidRenorm=6, MiniMax2=7, Sigmoid=8, TopKSigmoid=9, Unspecified=10
```
上游还为该枚举定义了可 `eval` 还原的 `__repr__`
（返回 `RoutingMethodType.Default` 而非 IntEnum 默认形式），注释说明「统一 MoE API 依赖它做配置往返」。

**做什么**：取值与成员集合完全对齐上游，补上 `__repr__`，修正 docstring。

#### T4.2 · `single_decode_with_kv_cache` 张量形状约定不同

**文件**：`flashinfer/decode.py:69`

**问题**：同名、同参数个数，但形状语义不同——
本仓 docstring 写 `q (B, H, D)`、`k/v_cache (B, S, H_kv, D)`；
上游（`flashinfer-ori/flashinfer/decode.py:908-912`）是
`q: [num_qo_heads, head_dim]`、`k: [kv_len, num_kv_heads, head_dim]`（**无 batch 维**）。

**做什么**：改成与上游一致的无 batch 语义；若确实要保留带 batch 的变体，
**另起一个不同的函数名**，不要占用上游的名字。

#### T4.3 · MoE 三个「兼容别名」实际改了参数名

**文件**：`flashinfer/fused_moe.py:396` 附近
```python
cutlass_fused_moe = fused_moe
trtllm_fp8_per_tensor_scale_moe = fp8_per_tensor_scale_moe
trtllm_fp8_block_scale_moe = fp8_block_scale_moe
```

**问题**：注释声称是 drop-in 兼容别名，但被指向的函数参数名改了
（如上游 `routing_logits` → 本仓 `expert_logits`），位置调用和关键字调用两条路都断。

**做什么**：要么让参数名与上游一致，要么删掉这些别名（别名存在却不兼容，比没有更糟）。

#### T4.4 · decode wrapper 的 `plan()` 参数集不同

**文件**：`flashinfer/decode.py` 中的 `BatchDecodeWithPagedKVCacheWrapper`

**问题**：`plan()` 参数**个数相同（7 个）但集合不同**——个数一样最容易被误认为兼容。
本仓丢了 `indptr` / `indices` / `last_page_len` 这三个分页 KV 的关键参数，
换成了 `q_dtype` / `kv_dtype` / `causal`。

另：上游该 wrapper 有 15 个公开方法，本仓 3 个。缺 `reset_workspace_buffer`（**vLLM 会调**）、
`forward`、`workspace_size`、`resolved_backend` 等。

**做什么**：`plan()` 参数集对齐上游；补上 `reset_workspace_buffer`（vLLM 依赖）。
**此任务与 T9 有重叠**，建议一起做。

#### T4.5 · 未知后端名抛裸异常

**文件**：`flashinfer/runtime/dispatch.py:91-93`
```python
policy = "auto" if backend == "auto" else Backend(backend)
```

**问题**：传上游的后端名（`"fa3"` / `"cudnn"` / `"trtllm-gen"`）得到
`ValueError: 'fa3' is not a valid Backend`，不说本仓支持哪些值。
这是移植用户最可能撞到的**第一个**错误，却给出最没用的提示。

**做什么**：捕获并重抛为带提示的 `DispatchError`，列出本仓支持的 backend 值，
并对已知的上游后端名给出「这是 CUDA 专有后端，昇腾上请用 X」的指引。

#### T4.6 · `enable_pdl` 行为在本仓内部自相矛盾 ⚠ 新增

**背景（重要，不要改错方向）**：
`enable_pdl` 是 **CUDA Programmatic Dependent Launch**（Hopper sm_90+），
让后一个 kernel 提前开始前导工作以隐藏启动延迟。**上游是真实实现它的**：
2,522 处引用、263 个文件、`@register_custom_op` 一路传到 kernel。

**上游自己的语义**（`flashinfer-ori/flashinfer/norm/__init__.py:186-187`
与 `flashinfer/utils.py:869-873`）：
```python
if enable_pdl is None or enable_pdl:
    enable_pdl = device_support_pdl(input.device)

def device_support_pdl(device) -> bool:
    if device.type != "cuda": return False       # ← 非 CUDA 设备直接 False
    major, _ = get_compute_capability(device)
    return major >= 9
```
**也就是说：上游在不支持 PDL 的硬件上会把 `enable_pdl=True` 静默降级为 False，不报错。**
`device_support_pdl` 里本来就有一条 `device.type != "cuda"` 的分支。

**结论**：`flashinfer/norm.py` 的 `del enable_pdl  # NPU 无 PDL` 是**与上游行为一致**的，
**不是「有意不同」**，账本里应标 `compatible` 而非 `intentionally_different`。

**问题所在**：`flashinfer/attention/batch.py:351-352` 却是
```python
if enable_pdl not in (None, False):
    raise NotImplementedError("enable_pdl is CUDA-specific and unsupported")
```
而 `tests/test_batch_frontend.py:293-294` 把这个 raise 固化成了期望行为
（`wrapper.run(q, kv, enable_pdl=True)` 抛 `NotImplementedError`）。
**同一个参数在本仓有两种相反的处理方式**，且 attention 这条**破坏了对标**——
上游在 A100 上跑 `enable_pdl=True` 不会抛错。

**做什么**：
1. 统一为**上游语义**：接受 `enable_pdl`，在 NPU 上等价于 False，不报错。
2. 改 `flashinfer/attention/batch.py` 去掉该 raise；相应更新
   `tests/test_batch_frontend.py:293-294`。
3. 把这条处理方式写进贡献规范：**上游有、昇腾无对应硬件能力的参数，一律保留参数位 +
   显式丢弃 + 注释说明**，账本标 `compatible`。
4. 注意区分：上游**确实会抛错**的场景是「某个特定后端结构上不支持 PDL」
   （如 `mla/_core.py:2183` 的 `backend='cute-dsl'`，抛 `ValueError`），
   那是后端级限制，不是硬件级降级。本仓若要抛，也只在这种情况下抛，且用 `ValueError`。

**T4 整体验收**：
```bash
pytest tests/ -q -k "parity or enum or pdl or backend"
# 并逐条人工核对本文件列出的 6 项
```

---

### T5 · 让对标账本可复现、可运行、覆盖完整

**依赖**：T1（`parity.py` 目前因 `SchemaError` 缺失而打不开）

**问题**：
- 账本记的基线是「某月某日的 main 快照」**而不是 commit**；
  两份账本用的还是相差 20 天的两个基线
  （`api_parity.json` 写 2026-08-05，`attention_api_parity.json` 写 2026-08-25）
- `api_parity.json` 只有 33 条，覆盖上游接口的 15%
- 校验命令跑不起来

**做什么**：
1. 基线改成**具体 commit hash**（`289557772`），两份账本统一。
2. **把 147 条覆盖表灌进去当初始内容**——信息量远大于现有的 33 条。
   每条带：上游接口名 / CANN 覆盖情况 / CANN 算子名 / 本仓状态 / vLLM·SGLang 是否在用。
3. 让 `flashinfer/parity.py` 的校验命令能跑通并进 CI。

**验收**：
```bash
python3 -m flashinfer parity-check   # 或仓内实际的命令名
# 期望：正常输出报告，退出码 0
python3 -c "
import json
d=json.load(open('flashinfer/data/api_parity.json'))
print('条目:', len(d.get('entries', d)))
"
# 期望 ≥ 147
```

---

# P2 · 地基扩展（登记表字段）

> **四件事里有三件卡在同一个地方**：登记表能描述的东西太少。
> 这不是某个功能的缺陷，是公共地基的缺陷。

### T6 · 给 `KernelDescriptor` / `CapabilitySpec` 加字段

**依赖**：T1

**文件**：`flashinfer/runtime/schema.py`、`flashinfer/runtime/hardware.py`、
`flashinfer/runtime/registry.py`、`flashinfer/runtime/dispatch.py`

**现状**：
```python
@dataclass(frozen=True)
class KernelDescriptor:
    kernel_id: str
    backend: Backend
    capability: CapabilitySpec
    compile_mode: CompileMode = CompileMode.NONE
    priority: int = 0
    runner: Optional[KernelRunner] = None     # ← 只能是 Python 函数
```
```python
@dataclass(frozen=True)
class CapabilitySpec:
    op: str
    dtypes / layouts / soc_models / min_aicore / specializations   # ← 无任何 shape 字段
```

**按以下顺序加**（顺序重要，前者是后者的前提）：

#### T6.1 · 执行体字段 — 解锁 AOT/JIT 与 csrc 下沉

加 `artifact: Optional[ArtifactRef]`（产物路径 + 入口符号 + 格式）
与 `source: Optional[SourceRef]`（语言 + 源码路径 + 哈希）。
同时需要 **SOC 型号作为编译维度**（同一 kernel 在 910B / 910_93 / 310P 上是不同产物）。

没有这个字段，**自定义算子编出来也无处登记**，vllm-ascend 的 `csrc/` 也搬不过来（T13）。

#### T6.2 · 「外部仓」身份类别 — 解锁外部高性能实现路由

**问题**：现在 `Backend` 只有 5 个值，把实现二分为「内置（默认参与自动选择）」和
「用户注册（默认**不**参与）」。而**外部高性能仓两边都不是**——
它不随本仓发布，但也不是用户临时塞进来的，是本仓主动选择、验证过、认为更快才对接的。

今天只能二选一，两条都不对：
- 登记成 `cann` → 语义造假，且被排序规则钉在固定档位
- 登记成 `custom` → **默认不参与自动选择**（`FLASHINFER_NPU_ALLOW_CUSTOM_KERNELS` 默认 `"0"`），
  一个一等产品功能默认是关闭的

**做什么**：新增 `Backend.EXTERNAL`（或等价身份字段），**默认参与 auto 候选池**，
与 `custom`（用户临时注册、需 opt-in）区分。
`register_custom()` 对 `custom/` 前缀的强制保留给 `custom`，外部仓用自己的命名空间。

**注意**：设计文档 §10.1 的接入示例用的是 `mycompany/...` 前缀，
而代码强制字面 `custom/` —— 照文档抄会抛错。此任务要顺便修正文档。

#### T6.3 · shape / 规模区间字段

现在**任何 head_dim、page_size 都能通过筛选**，然后在底层算子内部失败，
违反了设计文档自己定的「所有降级必须可见」。

三件事同时建立在这个字段上：按规模分档调优、图模式按 bucket 固化选择、选择结果的正确性。

#### T6.4 · 昇腾物理属性

- `CapabilitySpec` 加**物理排布**字段（NZ 分形 / ND）——现在没地方说明
- `HardwareSignature` 加 **UB（统一缓冲区）容量**——写 Ascend C 算子时决定 tiling 的首要约束

#### T6.5 · 入图能力声明 — 解锁图模式

加 `graph_capability` 字段，三选一：

| 值 | 含义 |
|---|---|
| `capture_safe` | 捕获即可用，重放不需要任何干预 |
| `needs_param_update` | 需要在重放前改写已录制任务的参数（tiling 随 seq_len 变） |
| `not_capturable` | 不能入图，该层必须留在图外（对应 PIECEWISE 模式） |

对 `needs_param_update`，记录自己要交代**「改哪些参数、怎么改」**——
由框架在重放前统一执行。

> 参考：vllm-ascend 今天是用 `use_updatable_graph(attn_backend)` 做这个判断，
> 实现是 `issubclass(attn_backend, AscendAttentionBackend)` —— **靠类型判断猜能力**。
> 这个字段就是要取代它。

**T6 验收**：
```bash
python3 -c "
from flashinfer.runtime.schema import KernelDescriptor, CapabilitySpec
import dataclasses
print([f.name for f in dataclasses.fields(KernelDescriptor)])
print([f.name for f in dataclasses.fields(CapabilitySpec)])
"
pytest tests/ -q
```

---

### T7 · 重做 dispatch 排序规则

**依赖**：T6.1、T6.2

**文件**：`flashinfer/runtime/dispatch.py:57-69`

**现状**：
```python
def _heuristic_tier(descriptor):
    if descriptor.backend == Backend.REFERENCE: return 3
    if descriptor.backend == Backend.CANN: return 1
    return 0 if descriptor.compile_mode == CompileMode.AOT else 2
```

**两个问题**：
1. 写死「自研 AOT 优于 CANN」—— 与「路由到外部仓/CANN 是因为它更快」这个前提冲突，
   在任何实测之前就断定了结论。
2. **外部仓的预编译产物会掉进 tier 2**（`else` 分支），**排在 CANN 后面**——
   既不符合语义，也与「它是已知更快的实现」相反。

**做什么**：重新设计档位，给外部仓留位置；
让「谁更快」尽量来自实测数据而非写死的假设，无数据时的默认顺序要能被配置覆盖。

---

# P2+ · 选型与编译的分层

> 这三条补的是「JIT 编译一整套」和「选取性能最优实现」这两个诉求里
> **前面阶段没覆盖到**的部分。T6 加的是**字段**，这里加的是**分层结构**。

### T18 · 统一的多后端 JIT 触发层，明确 kernel ↔ 产物/符号映射

**依赖**：T1、T6.1

**要解决的问题**：编译产物**不只有 Ascend C 一种**。Triton（triton-ascend）、
Ascend C、外部仓的预编译产物、未来可能的其它语言——**触发编译的方式各不相同，
但底层都是 kernel**。需要一层统一抽象，否则每接一种后端就要在 dispatch 里加一条分支，
回到上游「636 处 `if backend ==`」的老路。

**现状**：
- `flashinfer/jit/` 共 7 个文件 1,146 行，**只有 `cache_key.py`（63 行）能 import**，
  其余全部卡在 `SchemaError` / `ArtifactFormat` 缺失上
- `JitModuleLoader`（`jit/loading.py`）是个**纯 `Protocol`，一行实现都没有**
- 全仓 `subprocess` 命中 0 处（**无法拉起编译器**）、`ctypes` 命中 0 处（**无法装载 .so**）
- `jit/core.py` 全是元数据构造，没有一行触发编译

**上游对应物**（可作为形状参考，`flashinfer-ori/flashinfer/jit/`，**111 个文件、40,888 行**）：
- `JitSpec` —— 一份「要编什么」的声明，每个后端有自己的 `gen_jit_spec`，
  但**全部归约到同一个 `build_and_load()` 入口**
- `JitSpecRegistry` —— 全局登记所有 JitSpec
- `JitSpecStatus` —— 统一描述产物状态：
  `name` / `is_compiled` / `library_path` / `sources` / `needs_device_linking`

**做什么**：
1. 定义统一的「编译单元」抽象（对标 `JitSpec`）：声明源码、目标 SOC、编译选项、
   产物期望路径与入口符号。
2. **每个后端只提供「如何生成这份声明」和「如何触发自己的编译器」两个钩子**，
   其余（缓存、锁、原子替换、加载、登记）由统一层承担。
3. 明确 **kernel_id ↔ 产物文件 ↔ 入口符号**的三元映射，并让它可查询
   （类似 `JitSpecStatus`，能回答「这个 kernel 现在编了没有、产物在哪、符号叫什么」）。
4. 实现真正的编译触发（`subprocess`）与产物装载（`ctypes` 或 torch 扩展加载）。

**注意**：设计文档里关于编译缓存、文件锁、原子替换的描述有 100 多行，
**对应实现是 0 行**。这些要在本任务一并落地，不要只做「能编出来」。

**验收**：
```bash
# 能查询任意 kernel 的产物状态
python3 -c "
from flashinfer.jit import describe_artifact   # 名字按实际实现
print(describe_artifact('<某个 jit kernel_id>'))
"
# 至少两种后端（Triton + Ascend C）走同一个入口完成「声明→编译→加载→登记」
pytest tests/ -q -k "jit"
```

---

### T19 · 两级选择：先选实现栈，再选栈内配置

**依赖**：T6.3（shape 区间字段）、T6.4、T18

**要解决的问题**：目标形态是

```
(dtype / shape / compute capability)  →  哪个实现栈（哪个仓、哪种语言）
                                      →  栈内哪个具体配置
```

**这是两级。本仓目前只有一级。**

**现状**：`dispatch()` 在一个**扁平候选集**上做「筛选 → 排序 → 取第一个」
（`flashinfer/runtime/dispatch.py:155-215`），没有「栈」这个层次，
更没有「栈内配置」这个层次。

三个输入里：

| 输入 | 现状 |
|---|---|
| dtype | ✅ `CapabilitySpec.dtypes` 已有 |
| compute capability | ⚠️ 有 `soc_models` + `min_aicore`；**UB 容量缺**（T6.4 补） |
| **shape** | ❌ **完全没有**，任何 head_dim / page_size 都能通过筛选（T6.3 补） |

**第二级（栈内配置）完全不存在。**
上游的对应物是 `flashinfer/autotuner/`（**4,110 行**）+
`flashinfer/tuning_configs/`（按具体 SKU 存的配置文件，如 `*_NVIDIA_B200.py`）。

**做什么**：
1. 在登记表里引入「实现栈」这个层次——一个栈 = 一个来源（CANN / 本仓 Ascend C /
   本仓 Triton / 某个外部仓），栈内可以有多份**同语义但不同配置**的实现
   （不同 tiling、不同 block 划分、不同特化参数）。
2. dispatch 拆成两步：先按 (dtype, shape, capability) 选栈，
   再在栈内按配置选具体实现。
3. **两级都要可解释**——`explain_dispatch` 要能分别说明「为什么选这个栈」
   和「为什么选栈内这个配置」。现有的逐条淘汰原因机制要扩展到两级。

**注意**：不要把「栈」和现有的 `Backend` 枚举等同。`Backend` 是来源分类，
而「栈」还要区分同一来源下的不同实现集合。T6.2 新增的「外部仓」身份是栈的一种。

---

### T20 · 性能数据链路：让「选最优」有数据可依

**依赖**：T19

**要解决的问题**：「选取性能最优 kernel 实现」**必须有实测数据**。
现在没有任何采集、存储、查询机制，排序完全靠写死的假设。

**现状（比看起来更空）**：
- `flashinfer/tuning/` **全部内容是 26 行**，只有一个
  `query_tuning_record(signature, hardware) -> Optional[str]`，
  函数体就一行 `return None`，docstring 写「恒返回 None（未命中）」
- 更关键：**`dispatch.py` 根本没有 import 它**。
  所谓「排序先查 tuning 记录」只存在于 `dispatch.py:7` 和 `:204` 两句注释里，
  **连调用点都不存在**
- 实际排序是 `_heuristic_tier` 写死的档位：
  自研 AOT(0) > CANN(1) > 自研 JIT(2) > reference(3)

**这个档位与「性能最优」不是一回事，某些情况下恰好相反**——
它在任何实测之前就断定「自研优于 CANN」，而路由到 CANN / 外部仓的前提正是「它更快」。
（T7 指出了这个冲突，但 T7 只重排档位，不解决「没有数据」这个根因。）

**做什么**：
1. **采集**：建立 benchmark 入口，对同一 (op, 输入签名, 硬件) 下的所有候选实测耗时。
2. **存储**：按 (op, workload bucket, capability 指纹) 存冠军记录。
   bucket 依赖 T6.3 的规模区间字段——**没有区间，性能数据无法索引**。
3. **查询**：真正把 `query_tuning_record` 接进 `dispatch`，让「tuning 优先」
   从注释变成代码路径。
4. **分发**：区分「发版期实测数据库」（随包分发，对标上游 `tuning_configs/`）
   与「用户本地 tuning 库」（运行期产生），并明确两者的优先级。
5. 未命中时仍落确定性启发式——**这条要保留**，保证无数据时行为可复现。

**验收**：
```bash
# 有数据时 dispatch 选中的是实测最快的那个，且 explain 能说明依据
python3 -c "
import flashinfer
print(flashinfer.explain_dispatch(...))   # 期望出现「依据实测记录」而非「确定性启发式」
"
```

---

# P3 · 图模式（aclgraph）

> **路线已定**：按 aclgraph 做，torchair 最多留兼容。
> 依据：vllm-ascend 已把 aclgraph 作为默认图模式，torchair 标记废弃、v0.12.0rc1 移除、
> 最晚 2026 Q1。该仓 aclgraph 覆盖 20+ 文件，torchair 只剩一段 `ImportError` 兜底。
>
> **核心原则：本仓不捕获图，只负责让自己「可以被捕获」。** 与上游一致。

### 责任划分（动手前先读这张表）

| 动作 | 上游 GPU | 本仓昇腾 |
|---|---|---|
| 调 capture / replay | vLLM | **vllm-ascend，不下沉** |
| 决定捕获哪些 batch size、分桶 | vLLM | **vllm-ascend，不下沉** |
| 分配并持有固定 buffer | vLLM 分配后传入 | **vllm-ascend，不下沉** |
| plan/run 划分、冻结校验 | **FlashInfer** | **本仓**（照搬上游） |
| 最大 workspace 查询 | 不需要 | **本仓**（plan 期，新增） |
| 重放前改写 tiling 参数 | 不需要 | **本仓**（新增，GPU 无对应物） |

**捕获的范围是整个模型 forward**（embedding + N 层 + norm + 采样），
本仓只看得见一个算子，**结构上不可能也不应该**去捕获。

**但最后两行必须下沉**，因为它们是**算子级知识**：哪个算子、第几个参数位、
装的是什么 tiling、每步怎么变。今天它待在 vllm-ascend 的 attention 后端里，
**只是因为算子调用点在那儿**。若调用点下沉而这段不跟着下沉，
就变成「本仓发起调用、vllm-ascend 却要知道它的参数布局才能去补」——
一个算子的知识被劈在两个仓里，比现状更糟。

---

### T8 · 建立 torch.library 注册层与 meta 实现

**依赖**：T1

**为什么要做**：本仓**全仓 0 处** `torch.library` / `register_custom_op`。
对比：
- 上游有完整脚手架：`register_custom_op` / `register_fake_op` 定义在 `flashinfer/utils.py`，
  **194 个装饰点、128 个 fake op**（当前出于开销考虑短路成 `lambda x: x`，
  但结构完整，打开只是一行的事）
- vllm-ascend 有 **1,899 行** `torch_binding_meta.cpp` 专门写 meta 实现

**后果**：没有注册和 meta 实现，本仓的算子**无法被 torch.compile 追踪**，
在 PIECEWISE 图模式下会造成 graph break；也无法参与任何需要 fake tensor 推导的流程。

**做什么**：
1. 在 `flashinfer/utils.py` 建立与上游同名的 `register_custom_op` / `register_fake_op`
   （对标：同名同签名，便于后续代码直接对拍）。
2. 给已接入的算子加装饰与 meta（只推形状/dtype，不做计算、不分配设备内存）。
3. 验证 `torch.compile` 下不产生 graph break。

**验收**：
```bash
python3 -c "
import torch, flashinfer
# 对一个已接入算子做 torch.compile，检查无 graph break
"
TORCH_LOGS=graph_breaks python3 <你的最小复现脚本>
```

---

### T9 · 把图契约接到 `plan()` 上（上游那一半）

**依赖**：T1

**文件**：`flashinfer/attention/execution_identity.py`、
`flashinfer/attention/batch.py`、`flashinfer/decode.py`、`flashinfer/prefill.py`

**好消息：方向已经对了，形状也已经有了。**
`execution_identity.py:95` 的 `AttentionGraphResourceContract` 已声明
`graph_enabled` / `fixed_batch_size` / `persistent_buffers`，
并校验「开了图就必须有固定 batch、固定 buffer，且都在同一设备上」。
这是上游那一半的忠实移植。

**缺的是把它真正接到 `plan()` 上。**

**做什么**（逐条对标上游 `flashinfer-ori/flashinfer/decode.py`）：
1. wrapper **构造时**收固定 buffer（上游是 `paged_kv_indptr_buffer` /
   `paged_kv_indices_buffer` / `paged_kv_last_page_len_buffer` 三个）。
   本仓该构造函数目前只有 2 个参数，上游有 9 个。
2. `plan()` 只往固定 buffer 里 `copy_`，**不重新分配**，地址永不变。
3. **冻结语义**：batch size、后端选择、`q_len_per_req` 一旦首次捕获就冻住，
   下次 `plan()` 换了值**直接报错**，不要悄悄换 kernel——那会让已录的图对不上。
   上游的冻结检查覆盖 batch size、`q_len_per_req`、`backend="auto"` 的解析结果、
   block table 宽度，**全部「变了就抛错」**。
4. 删掉 `flashinfer/decode.py:158` 那句误导性注释：
   ```python
   del workspace_buffer, is_cuda_graph_compatible  # NPU 无对应机制，接受并忽略
   ```
   **aclgraph 就是那个对应机制。** 这句话会让后面接手的人以为这件事不需要做。

**命名**：对外参数名建议保留上游的 `use_cuda_graph` / `is_cuda_graph_compatible`
以维持 drop-in（内部是 backend-neutral 的 graph 标志，
本仓 `attention/batch.py:234` 已有 `is_cuda_graph_enabled` 作为兼容别名，沿用这个模式）。

**验收**：构造 wrapper 时传固定 buffer → 两次 `plan()` 用不同 batch size → 第二次抛错。

---

### T10 · plan 期的最大 workspace 查询

**依赖**：T9

**为什么**：昇腾算子（`npu_fused_infer_attention_score` 等）的 **workspace 大小依赖
每步变化的 seq_len**，而重放时没有 host 参与、算不了。
解法是捕获前问一个「所有 shape 都够用」的最大值，一次分好，地址不再变。

**参考实现**：vllm-ascend 用
`torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)`
（`vllm_ascend/attention/attention_v1.py:698`、`:770` 等 9 处）。
v2 版算子对应 `_npu_fused_infer_attention_score_v2_get_max_workspace`。

**做什么**：这件事**本来就属于 `plan()`**——「提前算好、固定住」正是 plan 的职责，
落在上游已有的 plan/run 分工里，不需要引入新概念。
在 `plan()` 中按图模式开关决定是否走最大 workspace 路径。

**代价要说明**：会多占显存。需要提供开关让调用方权衡。

---

### T11 · 通用的「重放前参数更新」机制 ⭐ 本阶段核心

**依赖**：T6.5（入图能力字段）、T10

**这是本仓相对上游要多承担的那一层，也是从 vllm-ascend 接手的核心资产。**

**vllm-ascend 现状**（要接手的东西）：
- `update_graph_params` **在 8 个 attention 后端里各写了一份**：
  `mla_v1` / `attention_v1` / `sfa_v1` / `dsa_v1` / `dsa_v41` /
  `mla_cp` / `dsa_cp` / `msa_m3`（共 9 处定义）
- 每份都手工把十几个参数**按位置解包、改掉、再塞回去**
- 两处带着作者自己的注释：
  ```
  # vllm_ascend/attention/mla_v1.py:954
  # FIXME: Behold! We are using a temporary hack here to update the args
  #        for each layer's attention op in the graph.
  ```

**有利的时间点**：vllm-ascend **自己也在收敛这件事**——
新写了通用的 `UpdatableGraph`（`vllm_ascend/compilation/updatable_graph.py`，217 行），
把「记句柄 → 重放前改参数」抽成框架能力，已有 12 个文件在用，
`attention_v1` 甚至把自己的手写版改成了「请走 UpdatableGraph」的报错。
**但迁移只做了一半**（`mla_v1` / `mla_cp` 仍在直接手写）。
**现在接手比等他们迁完再搬更省事。**

**底层 API**（CANN / torch_npu 公开能力）：
- `torch.npu.graph_task_update_begin(update_stream, handle)` / `graph_task_update_end()`
  —— 把新参数写回已录制的任务
- `torch.npu.ExternalEvent` —— 保证「改完再放」的顺序
- `torch_npu._C._NPUTaskGroupHandle` —— 捕获时记下的任务句柄

**做什么**：
1. 在 runtime 层实现一个通用机制：捕获期记录任务句柄与参数布局，
   重放前按 `graph_capability` 字段统一执行更新。
2. 由 `KernelDescriptor` 自己交代「要改哪些参数、怎么改」，
   **不要让每个算子各写一份**。
3. 对外暴露给 vllm-ascend 调用的接口：
   - 查询某个 kernel 的入图性质（取代 `issubclass` 判断）
   - 在 replay 前执行更新的回调

**注意：这会让本仓多出上游没有的 API 面。**
T10、T11 两项在上游**根本不存在对应接口**。
处理方式：放进明确的 NPU 扩展命名空间，**在对标账本里登记为「有意不同 / NPU-only 扩展」**，
不要让它看起来像上游 drop-in 接口的一部分，否则下游拿本仓当上游替身用时会撞上差异。

**验收**：一个 FULL 模式 aclgraph 捕获 + 多次不同 seq_len 的重放，结果与 eager 一致。

---

### T12 · 重写设计文档 §3.4

**依赖**：T9~T11 的设计确定后

**文件**：`docs/framework_design.md` §3.4（第 233-400 行，7 个小节，约 160 行）

**问题**：这 160 行设计的是 **torchair / GE 那条路**，核心机制是
「动态 tiling（host 回调）…图执行期每 step 重算 tiling + workspace size」。
**全文 0 次提到 aclgraph。**

**而这恰恰是 aclgraph 不允许的**——重放时没有 host 参与。
所以这不是「还没写到 aclgraph」，是**朝另一个方向写的**，要重写而非补充。

**做什么**：按 T9/T10/T11 的实际设计重写，明确：
- 路线选择 aclgraph 及其理由（含 torchair 废弃时间线）
- 责任划分（上面那张表）
- 入图能力声明的三种取值及语义
- 与登记表字段的对应关系

---

# P4 · 自定义算子与 vllm-ascend 下沉

> **终态**：vllm-ascend 从「自己持有 66 个 CANN 算子调用点 + 一套算子工程 + 8 份入图补丁」
> 变成**只调本仓接口**；本仓从「对标用的壳」变成**真正持有实现的那一层**。

### T13 · 把 vllm-ascend 的 `csrc/` 编译框架搬过来

**依赖**：T6.1（登记表能装编译产物）—— **顺序不能反**，
否则编出来的东西无处登记、选择器也选不到。

**为什么能搬**：`csrc/` 本来就是一个**自包含的 CANN 算子工程**，不依赖 vLLM 任何概念——
输入是 Ascend C 源码加一个 SOC 型号，输出是算子包和一个 `.so`。
整体搬过来，vllm-ascend 侧不会有业务逻辑受影响。

**为什么该搬**：本仓今天**连一行 Ascend C 源码、一条编译链路都没有**
（`kernels/ascendc` 是空目录，全仓 `subprocess` 0 处、`ctypes` 0 处），
而那边是一套已经在生产里跑的完整实现。**这不是「多接一个仓」，是把缺的地基直接拿过来。**

**要搬的东西**：

| 内容 | 规模 |
|---|---|
| Ascend C 算子源码 | `csrc/` 下 **1,179 个 cpp/h**，标准 CANN 工程布局（`op_host` / `op_kernel` 成对） |
| 构建脚本 | `csrc/build.sh` —— 按 `SOC_VERSION` 产出 `ophost / opapi / opgraph / onnxplugin` 四类目标；含 `ENABLE_BUILT_JIT` / `ENABLE_BUILT_CUSTOM` 开关 |
| aclnn 构建 | `csrc/build_aclnn.sh` |
| CMake 集成 | `csrc/CMakeLists.txt`、`setup.py` 的 `CMakeExtension` / `cmake_build_ext` |
| 第三方依赖 | catlass（git submodule） |

**SOC 型号维度**（`setup.py` 里的实际取值，T6.1 的字段要能表达这些）：
`ascend910b1`（A2）/ `ascend910_9391`（A3）/ `ascend310p1` / `ascend950*`（A5）

**做什么**：
1. 搬运目录结构与构建脚本，接进本仓的 `setup.py`。
2. 让编译产物能按 T6.1 的字段登记进登记表。
3. 打通 AOT（发版前按 SOC 出包）与 JIT（用到时现编）两条路径。

---

### T14 · 搬运算子注册层（torch.library + meta）

**依赖**：T8、T13

**vllm-ascend 现状**：
- `csrc/torch_binding.cpp`：**87 个 `ops.def` + 88 个 `ops.impl`**，
  注册在 `TORCH_LIBRARY_EXPAND(_C_ascend, ops)`，实现挂 `torch::kPrivateUse1`
- `csrc/torch_binding_meta.cpp`：**1,899 行** meta 实现（图模式追踪必需）
- Python 侧：`vllm_ascend/ops/register_custom_ops.py` 等处的
  `direct_register_custom_op` / `torch.library.register_fake`

**做什么**：把这层一并搬过来，与 T8 建立的注册层合并。
这层决定下沉后两边怎么对接。

---

### T15 · 下沉通用计算算子

**依赖**：T13、T14

**范围（这类没有疑问，先做）**：进出就是张量、与推理框架无关的算子。
- `grouped_matmul_swiglu_quant`（及 `_v2` / `_weight_nz_tensor_list` 变体）
- `dequant_swiglu_quant` / `swiglu_group_quant`
- `add_rms_norm_bias` / `rms_norm_cast`
- `bgmv_shrink` / `bgmv_expand` / `sgmv_shrink` / `sgmv_expand`（LoRA）
- `dequant_situ_quant` / `situ_mx_quant` / `gmm_dequant_situ_quant`

**做法**：下沉一个，在本仓登记一个，vllm-ascend 侧改成调本仓接口，删掉自己那份。

---

### T16 · 剥离带 vLLM 语义的融合算子，之后再下沉

**依赖**：T15

**范围（这类要小心）**：参数里裹着 vLLM 调度概念的算子——
`mla_preprocess`、`quant_lightning_indexer`、`kda_*`、
`fused_sparse_attention_overlap`、`sparse_flash_attention` 等。
它们的签名里有 block_table、slot_mapping、各种 metadata 布局，有的本来就是为某个模型写的。

**判据（可直接使用）**：
> **如果一个算子的签名里出现了只有 vLLM 才有的概念，它就还不该下沉。**
> 要下沉，先把那层概念剥掉，变成纯张量接口。

**依据不是额外要求，正是对标本身**：上游 FlashInfer 的 paged attention
只认 indptr / indices 这类通用结构，**不认任何引擎的调度对象**。

**做什么**：逐个算子评估 → 设计纯张量签名 → 在本仓实现 → vllm-ascend 侧做薄适配层。

---

# P5 · 收尾

### T17 · 处置 `attention/` 那两万多行

**依赖**：T1（在模块能打开之前，任何「该留该删」的判断都是猜测）、T6.2

**现状**：`flashinfer/attention/` 共 52 个模块、约 2 万行，**今天全部打不开**。
它定义了 `AttentionOperatorPackageRuntimeSpec`（18 个字段），其中
`supported_package_versions` / `adapter_version` / `observed_environment`
**正是对接一个有版本、会演进的外部仓所必须的**——方向是对的。

**问题在形态**：它把这些堆进一个 attention 专属的巨型结构，
在主干旁边另起了一套对象模型，而不是把「外部包」这个**身份**作为字段加回主登记表。

**合理终局**：
> 把「外部包适配」的概念**上移进主干登记表**（T6.2 已经在做这件事），
> attention 本身退回成和 norm、gemm 同构的一个普通算子域——
> 而不是一个拥有专属架构、却唯独跑不通的域。

**该保留的是它的设计意图，不是它的代码形态。**

---

## 附：待核实项

以下几条在规划阶段未能逐条验证，**实现前先查 CANN 算子清单确认**：

| 项 | 要确认什么 |
|---|---|
| `npu_gemma_rms_norm` | 是否确实没有 Add 融合版本（本仓目前用 Triton 自研补的） |
| `top_k_top_p_sample_v2` | 是否支持 MinP（影响 T2.1 的 `min_p_sampling_from_probs`） |
| `npu_scatter_pa_kv_cache` | 确切签名（影响 T2.6 的 `append_paged_kv_cache`） |
| `mega_moe` | 量化路径覆盖范围 |
| `moe_gating_top_k` | 覆盖表里写这个名字，检索到的是 `npu_moe_gating_top_k_softmax`，接入时对确切签名 |

## 附：一个已经走通的完整案例

`norm.gemma_fused_add_rmsnorm` 是**全仓唯一**走完整条路的实例：

```
对标表发现缺口 → 自研算子补上 → 登记进表 → 选择器自动选中
   （对标）      （自定义算子）   （注册开放）    （地基）
```

CANN 没有 Add+Gemma 的融合版本，本仓用 Triton 写了一个补上并登记进表。
做法上还有一个细节值得沿用：**登记时不 import triton，真正执行才加载**，
所以没装 triton 的机器照样能启动和做选择。

**架构本身是通的，后面就是把这条路重复 N 遍。**
