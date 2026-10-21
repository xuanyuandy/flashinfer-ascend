# T4 · 修掉会静默出错的对标缺陷

| | |
|---|---|
| **阶段** | P1 · 接口对齐（第一步任务） |
| **执行序号** | 4 / 20 |
| **依赖** | — |
| **规模** | 中 |
| **状态** | `未开始` |

> 索引与全局约束见 [`../TASKS.md`](../TASKS.md)。
> 架构背景与论证见配套的架构说明文档。
> **动手前先读 `../TASKS.md` 的「全局约束」一节。**

## 进度

- [ ] **T4.1** MoE 路由枚举取值与上游不一致 ⚠ 最高危
- [ ] **T4.2** `single_decode_with_kv_cache` 张量形状约定不同
- [ ] **T4.3** MoE 三个「兼容别名」实际改了参数名
- [ ] **T4.4** decode wrapper 的 `plan()` 参数集不同
- [ ] **T4.5** 未知后端名抛裸异常
- [ ] **T4.6** `enable_pdl` 行为在本仓内部自相矛盾 ⚠ 新增

---

## 阶段背景（P1 · 接口对齐（第一步任务））

> 这是本仓的主线职责。依据是 147 条算子覆盖表与本仓登记表的三列联表。
> 当前：本仓实际接入 15 条（全表 10%）；CANN 侧「已覆盖」46 条。

---

## 任务正文

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

## 完成时填写

- **实际改动**：
- **验收结果**：
- **遗留问题**（不要顺手改，记在这里）：
- **提交**：`T4 <标题>`
